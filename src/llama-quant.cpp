#include "llama-impl.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <regex>
#include <thread>
#include <unordered_map>

// result of parsing --tensor-type option
// (changes to this struct must be reflected in tools/quantize/quantize.cpp)
struct tensor_type_option {
    std::string name;
    ggml_type type = GGML_TYPE_COUNT;
};

// tensor categorization - used to avoid repeated string matching in quantization logic.
// this is different from LLM_TN - we want broad categories, not specific tensor names per arch.
enum class tensor_category {
    TOKEN_EMBD,
    ATTENTION_Q,
    ATTENTION_V,
    ATTENTION_K,
    ATTENTION_QKV,
    ATTENTION_KV_B,
    ATTENTION_OUTPUT,
    FFN_UP,
    FFN_GATE,
    FFN_DOWN,
    OUTPUT,
    OTHER
};

enum class sensitivity_bucket {
    UNKNOWN,
    LOW,
    MEDIUM,
    HIGH,
};

enum class sensitivity_source {
    NONE,
    HEURISTIC,
    IMATRIX,
    IMATRIX_BLOCKS,
};

enum class layer_similarity_bucket {
    UNKNOWN,
    LOW,
    MEDIUM,
    HIGH,
};

enum anchor_reason {
    ANCHOR_REASON_NONE         = 0,
    ANCHOR_REASON_FIRST_LAYER  = 1 << 0,
    ANCHOR_REASON_FINAL_LAYER  = 1 << 1,
    ANCHOR_REASON_SCENE_CHANGE = 1 << 2,
};

static void zeros(std::ofstream & file, size_t n) {
    char zero = 0;
    for (size_t i = 0; i < n; ++i) {
        file.write(&zero, 1);
    }
}

static std::string remap_layer(const std::string & orig_name, const std::vector<int> & prune, std::map<int, std::string> & mapped, int & next_id) {
    if (prune.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const int blk = std::stoi(match[1]);
        std::string new_name = orig_name;

        if (mapped.count(blk)) {
            // Already mapped, do nothing
        } else if (std::find(prune.begin(), prune.end(), blk) != prune.end()) {
            mapped[blk] = "";
        } else if (blk < prune.front()) {
            mapped[blk] = std::to_string(blk);
            next_id = blk + 1;
        } else {
            mapped[blk] = std::to_string(next_id);
            ++next_id;
        }

        return mapped[blk].empty() ? mapped[blk] : new_name.replace(match.position(1), match.length(1), mapped[blk]);
    }

    return orig_name;
}

static std::string remap_imatrix(const std::string & orig_name, const std::map<int, std::string> & mapped) {
    if (mapped.empty()) {
        return orig_name;
    }

    static const std::regex pattern(R"(blk\.(\d+)\.)");
    if (std::smatch match; std::regex_search(orig_name, match, pattern)) {
        const std::string blk(match[1]);
        std::string new_name = orig_name;

        for (const auto & p : mapped) {
            if (p.second == blk) {
                return new_name.replace(match.position(1), match.length(1), std::to_string(p.first));
            }
        }
        GGML_ABORT("\n%s: imatrix mapping error for %s\n", __func__, orig_name.c_str());
    }

    return orig_name;
}

//
// helper functions for tensor name matching
//

static bool tensor_name_match_token_embd(const char * tensor_name) {
    return std::strcmp(tensor_name, "token_embd.weight") == 0 ||
           std::strcmp(tensor_name, "per_layer_token_embd.weight") == 0;
}

static bool tensor_name_match_output_weight(const char * tensor_name) {
    return std::strcmp(tensor_name, "output.weight") == 0;
}

//
// tensor categorization for quantization
//
// (this is different from LLM_TN - we want broad categories, not specific tensor names per arch)
//

static tensor_category tensor_get_category(const std::string & tensor_name) {
    if (tensor_name_match_output_weight(tensor_name.c_str())) {
        return tensor_category::OUTPUT;
    }
    if (tensor_name_match_token_embd(tensor_name.c_str())) {
        return tensor_category::TOKEN_EMBD;
    }
    if (tensor_name.find("attn_qkv.weight") != std::string::npos) {
        return tensor_category::ATTENTION_QKV;
    }
    if (tensor_name.find("attn_kv_b.weight") != std::string::npos) {
        return tensor_category::ATTENTION_KV_B;
    }
    if (tensor_name.find("attn_v.weight") != std::string::npos) {
        return tensor_category::ATTENTION_V;
    }
    if (tensor_name.find("attn_k.weight") != std::string::npos) {
        return tensor_category::ATTENTION_K;
    }
    if (tensor_name.find("attn_q.weight") != std::string::npos) {
        return tensor_category::ATTENTION_Q;
    }
    if (tensor_name.find("attn_output.weight") != std::string::npos) {
        return tensor_category::ATTENTION_OUTPUT;
    }
    if (tensor_name.find("ffn_up") != std::string::npos) {
        return tensor_category::FFN_UP;
    }
    if (tensor_name.find("ffn_gate") != std::string::npos) {
        return tensor_category::FFN_GATE;
    }
    if (tensor_name.find("ffn_down") != std::string::npos) {
        return tensor_category::FFN_DOWN;
    }
    return tensor_category::OTHER;
}

static int tensor_get_layer_index(const std::string & tensor_name) {
    int layer = -1;
    if (sscanf(tensor_name.c_str(), "blk.%d.", &layer) == 1) {
        return layer;
    }
    return -1;
}

// check if category is for attention-v-like tensors (more sensitive to quantization)
static bool category_is_attn_v(tensor_category cat) {
    return cat == tensor_category::ATTENTION_V     ||
           cat == tensor_category::ATTENTION_QKV   ||
           cat == tensor_category::ATTENTION_KV_B;
}

static const char * sensitivity_bucket_name(sensitivity_bucket bucket) {
    switch (bucket) {
        case sensitivity_bucket::LOW:     return "low";
        case sensitivity_bucket::MEDIUM:  return "medium";
        case sensitivity_bucket::HIGH:    return "high";
        case sensitivity_bucket::UNKNOWN: return "unknown";
    }
    return "unknown";
}

static const char * sensitivity_source_name(sensitivity_source source) {
    switch (source) {
        case sensitivity_source::HEURISTIC: return "heuristic";
        case sensitivity_source::IMATRIX:   return "imatrix";
        case sensitivity_source::IMATRIX_BLOCKS: return "imatrix-blocks";
        case sensitivity_source::NONE:      return "none";
    }
    return "none";
}

static const char * layer_similarity_bucket_name(layer_similarity_bucket bucket) {
    switch (bucket) {
        case layer_similarity_bucket::LOW:     return "LOW_SIMILARITY";
        case layer_similarity_bucket::MEDIUM:  return "MEDIUM_SIMILARITY";
        case layer_similarity_bucket::HIGH:    return "HIGH_SIMILARITY";
        case layer_similarity_bucket::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

static std::string anchor_reason_name(int reason) {
    std::string result;
    if (reason & ANCHOR_REASON_FIRST_LAYER) {
        result += "first_layer";
    }
    if (reason & ANCHOR_REASON_FINAL_LAYER) {
        if (!result.empty()) result += "+";
        result += "final_layer";
    }
    if (reason & ANCHOR_REASON_SCENE_CHANGE) {
        if (!result.empty()) result += "+";
        result += "scene_change";
    }
    return result.empty() ? "none" : result;
}

static sensitivity_bucket spqr_guided_heuristic_bucket(tensor_category category) {
    // SpQR-inspired sensitivity guidance only: this does not store sparse outliers
    // and does not change runtime tensor formats.
    switch (category) {
        case tensor_category::OUTPUT:
        case tensor_category::TOKEN_EMBD:
        case tensor_category::ATTENTION_OUTPUT:
        case tensor_category::FFN_DOWN:
            return sensitivity_bucket::HIGH;
        case tensor_category::ATTENTION_Q:
        case tensor_category::ATTENTION_K:
        case tensor_category::ATTENTION_V:
        case tensor_category::ATTENTION_QKV:
        case tensor_category::ATTENTION_KV_B:
        case tensor_category::FFN_GATE:
        case tensor_category::FFN_UP:
            return sensitivity_bucket::MEDIUM;
        case tensor_category::OTHER:
            return sensitivity_bucket::LOW;
    }
    return sensitivity_bucket::LOW;
}

static bool ggml_type_is_below_q4_for_policy(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return true;
        default:
            return false;
    }
}

static ggml_type spqr_guided_type_for_bucket(
        const llama_model_quantize_params * params,
        tensor_category category,
        ggml_type base_type,
        sensitivity_bucket bucket) {
    ggml_type selected = base_type;

    switch (bucket) {
        case sensitivity_bucket::HIGH:
            selected = (category == tensor_category::OUTPUT || category == tensor_category::TOKEN_EMBD) ? GGML_TYPE_Q6_K : GGML_TYPE_Q5_K;
            break;
        case sensitivity_bucket::MEDIUM:
            selected = GGML_TYPE_Q4_K;
            break;
        case sensitivity_bucket::LOW:
            selected = base_type;
            break;
        case sensitivity_bucket::UNKNOWN:
            break;
    }

    if ((category == tensor_category::OUTPUT || category == tensor_category::TOKEN_EMBD) &&
            ggml_type_is_below_q4_for_policy(selected)) {
        selected = GGML_TYPE_Q4_K;
    }

    if (params->output_tensor_type < GGML_TYPE_COUNT && category == tensor_category::OUTPUT) {
        selected = params->output_tensor_type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && category == tensor_category::TOKEN_EMBD) {
        selected = params->token_embedding_type;
    }

    return selected;
}

static bool tensor_category_participates_in_layer_delta(tensor_category category) {
    switch (category) {
        case tensor_category::ATTENTION_Q:
        case tensor_category::ATTENTION_K:
        case tensor_category::ATTENTION_V:
        case tensor_category::ATTENTION_QKV:
        case tensor_category::ATTENTION_KV_B:
        case tensor_category::ATTENTION_OUTPUT:
        case tensor_category::FFN_GATE:
        case tensor_category::FFN_UP:
        case tensor_category::FFN_DOWN:
            return true;
        default:
            return false;
    }
}

static ggml_type spqr_layer_delta_type_for_bucket(
        const llama_model_quantize_params * params,
        tensor_category category,
        ggml_type base_type,
        sensitivity_bucket sensitivity,
        layer_similarity_bucket similarity,
        bool is_anchor) {
    if (!tensor_category_participates_in_layer_delta(category)) {
        return spqr_guided_type_for_bucket(params, category, base_type, sensitivity);
    }

    ggml_type selected = base_type;
    if (sensitivity == sensitivity_bucket::HIGH && similarity == layer_similarity_bucket::LOW) {
        selected = GGML_TYPE_Q6_K;
    } else if (sensitivity == sensitivity_bucket::HIGH && similarity == layer_similarity_bucket::HIGH) {
        selected = GGML_TYPE_Q5_K;
    } else if (sensitivity == sensitivity_bucket::HIGH) {
        selected = GGML_TYPE_Q5_K;
    } else if (sensitivity == sensitivity_bucket::MEDIUM && similarity == layer_similarity_bucket::HIGH) {
        selected = base_type;
    } else if (sensitivity == sensitivity_bucket::MEDIUM) {
        selected = GGML_TYPE_Q4_K;
    } else if (sensitivity == sensitivity_bucket::LOW && similarity == layer_similarity_bucket::HIGH) {
        selected = base_type;
    } else if (sensitivity == sensitivity_bucket::LOW && similarity == layer_similarity_bucket::LOW) {
        selected = ggml_type_is_below_q4_for_policy(base_type) ? GGML_TYPE_Q4_K : base_type;
    } else {
        selected = base_type;
    }

    if (is_anchor) {
        if (sensitivity == sensitivity_bucket::HIGH) {
            if (selected == GGML_TYPE_Q4_K || ggml_type_is_below_q4_for_policy(selected)) {
                selected = GGML_TYPE_Q5_K;
            }
        } else if (ggml_type_is_below_q4_for_policy(selected)) {
            selected = GGML_TYPE_Q4_K;
        }
    }

    if (params->output_tensor_type < GGML_TYPE_COUNT && category == tensor_category::OUTPUT) {
        selected = params->output_tensor_type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && category == tensor_category::TOKEN_EMBD) {
        selected = params->token_embedding_type;
    }

    return selected;
}

//
// quantization state
//

struct quantize_state_impl {
    const llama_model                 & model;
    const llama_model_quantize_params * params;

    int n_attention_wv = 0;
    int n_ffn_down     = 0;
    int n_ffn_gate     = 0;
    int n_ffn_up       = 0;
    int i_attention_wv = 0;
    int i_ffn_down     = 0;
    int i_ffn_gate     = 0;
    int i_ffn_up       = 0;

    int n_fallback    = 0;

    bool has_imatrix = false;

    int n_spqr_low      = 0;
    int n_spqr_medium   = 0;
    int n_spqr_high     = 0;
    int n_spqr_promoted = 0;
    int n_spqr_block_scored = 0;
    int n_delta_low     = 0;
    int n_delta_medium  = 0;
    int n_delta_high    = 0;
    int n_delta_demoted = 0;
    int n_anchor_layers = 0;
    int n_anchor_tensors = 0;
    int n_rd_selected = 0;

    // used to figure out if a model has tied embeddings (tok_embd shares weights with output)
    bool has_tied_embeddings = true; // assume tied until we see output.weight

    // tensor type override patterns (compiled once, used twice)
    std::vector<std::pair<std::regex, ggml_type>> tensor_type_patterns;

    quantize_state_impl(const llama_model & model, const llama_model_quantize_params * params):
        model(model), params(params)
    {
        // compile regex patterns once - they are expensive
        if (params->tt_overrides) {
            for (const auto * p = params->tt_overrides; p->pattern != nullptr; p++) {
                tensor_type_patterns.emplace_back(std::regex(p->pattern), p->type);
            }
        }
    }
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    tensor_category category;
    sensitivity_bucket sensitivity;
    sensitivity_source sensitivity_from;
    float           sensitivity_score;
    int             block_low;
    int             block_medium;
    int             block_high;
    int             block_count;
    int             layer;
    float           rel_delta_norm;
    float           cosine_similarity;
    layer_similarity_bucket layer_similarity;
    int             anchor_reason;
    ggml_type       rd_type;
    float           rd_cost;
    float           rd_distortion;
    float           rd_bpw;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
};

//
// dequantization
//

static void llama_tensor_dequantize_impl(
    ggml_tensor * tensor, std::vector<no_init<float>> & output, std::vector<std::thread> & workers,
    const size_t nelements, const int nthread
) {
    if (output.size() < nelements) {
        output.resize(nelements);
    }
    float * f32_output = (float *) output.data();

    const ggml_type_traits * qtype = ggml_get_type_traits(tensor->type);
    if (ggml_is_quantized(tensor->type)) {
        if (qtype->to_float == NULL) {
            throw std::runtime_error(format("type %s unsupported for integer quantization: no dequantization available", ggml_type_name(tensor->type)));
        }
    } else if (tensor->type != GGML_TYPE_F16 &&
               tensor->type != GGML_TYPE_BF16) {
        throw std::runtime_error(format("cannot dequantize/convert tensor type %s", ggml_type_name(tensor->type)));
    }

    if (nthread < 2) {
        if (tensor->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t *)tensor->data, f32_output, nelements);
        } else if (tensor->type == GGML_TYPE_BF16) {
            ggml_bf16_to_fp32_row((ggml_bf16_t *)tensor->data, f32_output, nelements);
        } else if (ggml_is_quantized(tensor->type)) {
            qtype->to_float(tensor->data, f32_output, nelements);
        } else {
            GGML_ABORT("fatal error"); // unreachable
        }
        return;
    }

    size_t block_size;
    if (tensor->type == GGML_TYPE_F16 ||
        tensor->type == GGML_TYPE_BF16) {
        block_size = 1;
    } else {
        block_size = (size_t)ggml_blck_size(tensor->type);
    }

    size_t block_size_bytes = ggml_type_size(tensor->type);

    GGML_ASSERT(nelements % block_size == 0);
    size_t nblocks = nelements / block_size;
    size_t blocks_per_thread = nblocks / nthread;
    size_t spare_blocks = nblocks - (blocks_per_thread * nthread); // if blocks aren't divisible by thread count

    size_t in_buff_offs = 0;
    size_t out_buff_offs = 0;

    for (int tnum = 0; tnum < nthread; tnum++) {
        size_t thr_blocks = blocks_per_thread + (tnum == nthread - 1 ? spare_blocks : 0); // num blocks for this thread
        size_t thr_elems = thr_blocks * block_size; // number of elements for this thread
        size_t thr_block_bytes = thr_blocks * block_size_bytes; // number of input bytes for this thread

        auto compute = [qtype] (ggml_type typ, uint8_t * inbuf, float * outbuf, int nels) {
            if (typ == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((ggml_fp16_t *)inbuf, outbuf, nels);
            } else if (typ == GGML_TYPE_BF16) {
                ggml_bf16_to_fp32_row((ggml_bf16_t *)inbuf, outbuf, nels);
            } else {
                qtype->to_float(inbuf, outbuf, nels);
            }
        };
        workers.emplace_back(compute, tensor->type, (uint8_t *) tensor->data + in_buff_offs, f32_output + out_buff_offs, thr_elems);
        in_buff_offs += thr_block_bytes;
        out_buff_offs += thr_elems;
    }
    for (auto & w : workers) { w.join(); }
    workers.clear();
}

static void load_tensor_as_f32(
        llama_model_loader & ml,
        ggml_tensor * tensor,
        std::vector<no_init<uint8_t>> & read_data,
        std::vector<no_init<float>> & f32_conv_buf,
        std::vector<float> & out,
        std::vector<std::thread> & workers,
        int nthread) {
    const size_t tensor_size = ggml_nbytes(tensor);
    if (!ml.use_mmap) {
        if (read_data.size() < tensor_size) {
            read_data.resize(tensor_size);
        }
        tensor->data = read_data.data();
    }
    ml.load_data_for(tensor);

    const int64_t nelements = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        const float * data = (const float *) tensor->data;
        out.assign(data, data + nelements);
    } else {
        llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
        const float * data = (const float *) f32_conv_buf.data();
        out.assign(data, data + nelements);
    }
}

static layer_similarity_bucket layer_similarity_bucket_from_metrics(float rel_delta_norm, float cosine_similarity) {
    if (rel_delta_norm <= 0.10f || cosine_similarity >= 0.995f) {
        return layer_similarity_bucket::HIGH;
    }
    if (rel_delta_norm <= 0.25f || cosine_similarity >= 0.980f) {
        return layer_similarity_bucket::MEDIUM;
    }
    return layer_similarity_bucket::LOW;
}

static float percentile(std::vector<float> values, float p);

static void init_layer_delta_analysis(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        int nthread,
        bool print_report) {
    std::map<std::pair<int, int>, size_t> layer_category_to_index;
    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        tm.layer = tensor_get_layer_index(tm.name);
        if (tm.layer >= 0 && tensor_category_participates_in_layer_delta(tm.category)) {
            layer_category_to_index[{ tm.layer, (int) tm.category }] = i;
        }
    }

    std::vector<no_init<uint8_t>> read_data_prev;
    std::vector<no_init<uint8_t>> read_data_cur;
    std::vector<no_init<float>> f32_conv_prev;
    std::vector<no_init<float>> f32_conv_cur;
    std::vector<float> prev_data;
    std::vector<float> cur_data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::map<int, std::vector<float>> rel_by_layer;
    std::map<int, std::vector<float>> cos_by_layer;

    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        if (!tm.allows_quantization || tm.layer <= 0 || !tensor_category_participates_in_layer_delta(tm.category)) {
            continue;
        }

        auto prev_it = layer_category_to_index.find({ tm.layer - 1, (int) tm.category });
        if (prev_it == layer_category_to_index.end()) {
            continue;
        }

        ggml_tensor * cur_tensor = tensors[i]->tensor;
        ggml_tensor * prev_tensor = tensors[prev_it->second]->tensor;
        if (ggml_nelements(cur_tensor) != ggml_nelements(prev_tensor)) {
            continue;
        }

        load_tensor_as_f32(ml, prev_tensor, read_data_prev, f32_conv_prev, prev_data, workers, nthread);
        load_tensor_as_f32(ml, cur_tensor,  read_data_cur,  f32_conv_cur,  cur_data,  workers, nthread);

        double delta_norm2 = 0.0;
        double cur_norm2 = 0.0;
        double prev_norm2 = 0.0;
        double dot = 0.0;

        for (size_t j = 0; j < cur_data.size(); ++j) {
            const double cur = cur_data[j];
            const double prev = prev_data[j];
            const double delta = cur - prev;
            delta_norm2 += delta * delta;
            cur_norm2   += cur * cur;
            prev_norm2  += prev * prev;
            dot         += cur * prev;
        }

        const double eps = 1e-12;
        tm.rel_delta_norm = (float) (std::sqrt(delta_norm2) / (std::sqrt(cur_norm2) + eps));
        tm.cosine_similarity = (float) (dot / ((std::sqrt(cur_norm2) * std::sqrt(prev_norm2)) + eps));
        tm.layer_similarity = layer_similarity_bucket_from_metrics(tm.rel_delta_norm, tm.cosine_similarity);

        switch (tm.layer_similarity) {
            case layer_similarity_bucket::LOW:    ++qs.n_delta_low;    break;
            case layer_similarity_bucket::MEDIUM: ++qs.n_delta_medium; break;
            case layer_similarity_bucket::HIGH:   ++qs.n_delta_high;   break;
            case layer_similarity_bucket::UNKNOWN: break;
        }

        rel_by_layer[tm.layer].push_back(tm.rel_delta_norm);
        cos_by_layer[tm.layer].push_back(tm.cosine_similarity);

        if (print_report) {
            LLAMA_LOG_INFO("%s: layer-delta layer=%4d tensor=%-36s rel_delta_norm=%10.6f cosine_sim=%10.6f sensitivity=%-6s similarity=%s\n",
                    __func__,
                    tm.layer,
                    tm.name.c_str(),
                    tm.rel_delta_norm,
                    tm.cosine_similarity,
                    sensitivity_bucket_name(tm.sensitivity),
                    layer_similarity_bucket_name(tm.layer_similarity));
        }
    }

    double rel_sum = 0.0;
    double cos_sum = 0.0;
    int n_pairs = 0;
    std::vector<std::pair<int, float>> layer_rel;
    std::map<int, float> avg_rel_by_layer;
    for (const auto & kv : rel_by_layer) {
        const auto & rels = kv.second;
        const auto & coss = cos_by_layer[kv.first];
        const float avg_rel = std::accumulate(rels.begin(), rels.end(), 0.0f) / rels.size();
        const float avg_cos = std::accumulate(coss.begin(), coss.end(), 0.0f) / coss.size();
        layer_rel.push_back({ kv.first, avg_rel });
        avg_rel_by_layer[kv.first] = avg_rel;
        rel_sum += avg_rel;
        cos_sum += avg_cos;
        ++n_pairs;
    }

    if (qs.params->adaptive_anchors) {
        std::map<int, int> anchor_reasons;
        const int n_layer = (int) qs.model.hparams.n_layer;
        if (n_layer > 0) {
            anchor_reasons[0] |= ANCHOR_REASON_FIRST_LAYER;
            anchor_reasons[n_layer - 1] |= ANCHOR_REASON_FINAL_LAYER;
        }

        std::vector<float> layer_scores;
        layer_scores.reserve(avg_rel_by_layer.size());
        for (const auto & kv : avg_rel_by_layer) {
            layer_scores.push_back(kv.second);
        }

        float scene_threshold = 0.0f;
        if (!layer_scores.empty()) {
            scene_threshold = percentile(layer_scores, qs.params->anchor_percentile / 100.0f);
            for (const auto & kv : avg_rel_by_layer) {
                if (kv.second >= scene_threshold) {
                    anchor_reasons[kv.first] |= ANCHOR_REASON_SCENE_CHANGE;
                }
            }
        }

        for (auto & tm : metadata) {
            if (tm.layer < 0 || !tensor_category_participates_in_layer_delta(tm.category)) {
                continue;
            }
            auto it = anchor_reasons.find(tm.layer);
            if (it != anchor_reasons.end()) {
                tm.anchor_reason = it->second;
                ++qs.n_anchor_tensors;
            }
        }
        qs.n_anchor_layers = (int) anchor_reasons.size();

        LLAMA_LOG_INFO("%s: adaptive anchors: layers=%d tensors=%d percentile=%.1f scene_threshold=%.6f\n",
                __func__, qs.n_anchor_layers, qs.n_anchor_tensors, qs.params->anchor_percentile, scene_threshold);
        if (qs.params->print_anchor_report) {
            for (const auto & kv : anchor_reasons) {
                const auto score_it = avg_rel_by_layer.find(kv.first);
                LLAMA_LOG_INFO("%s: anchor layer=%4d avg_rel_delta=%10.6f reason=%s\n",
                        __func__,
                        kv.first,
                        score_it != avg_rel_by_layer.end() ? score_it->second : 0.0f,
                        anchor_reason_name(kv.second).c_str());
            }
        }
    }

    std::sort(layer_rel.begin(), layer_rel.end(), [](const auto & a, const auto & b) {
        return a.second > b.second;
    });

    LLAMA_LOG_INFO("%s: layer-delta summary: tensors high=%d medium=%d low=%d avg_layer_rel_delta=%10.6f avg_layer_cosine=%10.6f\n",
            __func__,
            qs.n_delta_high,
            qs.n_delta_medium,
            qs.n_delta_low,
            n_pairs > 0 ? rel_sum / n_pairs : 0.0,
            n_pairs > 0 ? cos_sum / n_pairs : 0.0);

    if (!layer_rel.empty()) {
        LLAMA_LOG_INFO("%s: layer-delta most unique layers:", __func__);
        for (size_t i = 0; i < std::min<size_t>(3, layer_rel.size()); ++i) {
            LLAMA_LOG_INFO(" %d(%.4f)", layer_rel[i].first, layer_rel[i].second);
        }
        LLAMA_LOG_INFO("\n");

        std::sort(layer_rel.begin(), layer_rel.end(), [](const auto & a, const auto & b) {
            return a.second < b.second;
        });
        LLAMA_LOG_INFO("%s: layer-delta most redundant-looking layers:", __func__);
        for (size_t i = 0; i < std::min<size_t>(3, layer_rel.size()); ++i) {
            LLAMA_LOG_INFO(" %d(%.4f)", layer_rel[i].first, layer_rel[i].second);
        }
        LLAMA_LOG_INFO("\n");
    }
}

//
// do we allow this tensor to be quantized?
//

static bool tensor_allows_quantization(const llama_model_quantize_params * params, llm_arch arch, const ggml_tensor * tensor) {
    // trivial checks first -- no string ops needed
    if (params->only_copy)       return false;

    // quantize only 2D and 3D tensors (experts)
    if (ggml_n_dims(tensor) < 2) return false;

    const std::string name = ggml_get_name(tensor);

    // This used to be a regex, but <regex> has an extreme cost to compile times.
    bool quantize = name.rfind("weight") == name.size() - 6; // ends with 'weight'?

    // do not quantize norm tensors
    quantize &= name.find("_norm.weight") == std::string::npos;

    quantize &= params->quantize_output_tensor || name != "output.weight";

    // do not quantize expert gating tensors
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ffn_gate_inp.weight") == std::string::npos;

    // these are very small (e.g. 4x4)
    quantize &= name.find("altup")  == std::string::npos;
    quantize &= name.find("laurel") == std::string::npos;

    // these are not too big so keep them as it is
    quantize &= name.find("per_layer_model_proj") == std::string::npos;

    // do not quantize positional embeddings and token types (BERT)
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_POS_EMBD,    "weight");
    quantize &= name != LLM_TN(arch)(LLM_TENSOR_TOKEN_TYPES, "weight");

    // do not quantize Mamba/Kimi's small conv1d weights
    // NOTE: can't use LLM_TN here because the layer number is not known
    quantize &= name.find("ssm_conv1d") == std::string::npos;
    quantize &= name.find("shortconv.conv.weight") == std::string::npos;

    // do not quantize RWKV's small yet 2D weights
    quantize &= name.find("time_mix_first.weight") == std::string::npos;
    quantize &= name.find("time_mix_w0.weight") == std::string::npos;
    quantize &= name.find("time_mix_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_v0.weight") == std::string::npos;
    quantize &= name.find("time_mix_v1.weight") == std::string::npos;
    quantize &= name.find("time_mix_v2.weight") == std::string::npos;
    quantize &= name.find("time_mix_a0.weight") == std::string::npos;
    quantize &= name.find("time_mix_a1.weight") == std::string::npos;
    quantize &= name.find("time_mix_a2.weight") == std::string::npos;
    quantize &= name.find("time_mix_g1.weight") == std::string::npos;
    quantize &= name.find("time_mix_g2.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w1.weight") == std::string::npos;
    quantize &= name.find("time_mix_decay_w2.weight") == std::string::npos;
    quantize &= name.find("time_mix_lerp_fused.weight") == std::string::npos;

    // do not quantize relative position bias (T5)
    quantize &= name.find("attn_rel_b.weight") == std::string::npos;

    // do not quantize specific multimodal tensors
    quantize &= name.find(".position_embd") == std::string::npos;
    quantize &= name.find("sam.pos_embd")   == std::string::npos;
    quantize &= name.find("sam.neck.")      == std::string::npos;
    quantize &= name.find("sam.net_")       == std::string::npos;
    quantize &= name.find(".rel_pos")       == std::string::npos;
    quantize &= name.find(".patch_embd")    == std::string::npos;
    quantize &= name.find(".patch_merger")  == std::string::npos;

    return quantize;
}

//
// tensor type selection
//

// incompatible tensor shapes are handled here - fallback to a compatible type
static ggml_type tensor_type_fallback(quantize_state_impl & qs, const ggml_tensor * t, const ggml_type target_type) {
    ggml_type return_type = target_type;

    const int64_t ncols = t->ne[0];
    const int64_t qk_k = ggml_blck_size(target_type);

    if (ncols % qk_k != 0) { // this tensor's shape is incompatible with this quant
        LLAMA_LOG_WARN("warning: %-36s - ncols %6" PRId64 " not divisible by %3" PRId64 " (required for type %7s) ",
                        t->name, ncols, qk_k, ggml_type_name(target_type));
        ++qs.n_fallback;

        switch (target_type) {
            // types on the left: block size 256
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ3_S:   // types on the right: block size 32
            case GGML_TYPE_IQ4_XS:  return_type = GGML_TYPE_IQ4_NL; break;
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_TQ1_0:
            case GGML_TYPE_TQ2_0:   return_type = GGML_TYPE_Q4_0;   break;
            case GGML_TYPE_Q4_K:    return_type = GGML_TYPE_Q5_0;   break;
            case GGML_TYPE_Q5_K:    return_type = GGML_TYPE_Q5_1;   break;
            case GGML_TYPE_Q6_K:    return_type = GGML_TYPE_Q8_0;   break;
            default:
                throw std::runtime_error(format("no tensor type fallback is defined for type %s",
                                                ggml_type_name(target_type)));
        }
        if (ncols % ggml_blck_size(return_type) != 0) {
            //
            // the fallback return type is still not compatible for this tensor!
            //
            // most likely, this tensor's first dimension is not divisible by 32.
            // this is very rare. we can either abort the quantization, or
            // fallback to F16 / F32.
            //
            LLAMA_LOG_WARN("(WARNING: must use F16 due to unusual shape) ");
            return_type = GGML_TYPE_F16;
        }
        LLAMA_LOG_WARN("-> falling back to %7s\n", ggml_type_name(return_type));
    }
    return return_type;
}

// internal standard logic for selecting the target tensor type based on tensor category, ftype, and model arch
static ggml_type llama_tensor_get_type_impl(quantize_state_impl & qs, ggml_type new_type, const ggml_tensor * tensor, llama_ftype ftype, tensor_category category) {
    const std::string name = ggml_get_name(tensor);

    // TODO: avoid hardcoded tensor names - use the TN_* constants
    const llm_arch arch = qs.model.arch;

    auto use_more_bits = [](int i_layer, int n_layers) -> bool {
        return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
    };
    const int n_expert = std::max(1, (int)qs.model.hparams.n_expert);
    auto layer_info = [n_expert] (int i_layer, int n_layer, const char * name) {
        if (n_expert > 1) {
            // Believe it or not, "experts" in the FFN of Mixtral-8x7B are not consecutive, but occasionally randomly
            // sprinkled in the model. Hence, simply dividing i_ffn_down by n_expert does not work
            // for getting the current layer as I initially thought, and we need to resort to parsing the
            // tensor name.
            if (sscanf(name, "blk.%d.", &i_layer) != 1) {
                throw std::runtime_error(format("Failed to determine layer for tensor %s", name));
            }
            if (i_layer < 0 || i_layer >= n_layer) {
                throw std::runtime_error(format("Bad layer %d for tensor %s. Must be in [0, %d)", i_layer, name, n_layer));
            }
        }
        return std::make_pair(i_layer, n_layer);
    };

    // for arches that share the same tensor between the token embeddings and the output, we quantize the token embeddings
    // with the quantization of the output tensor
    if (category == tensor_category::OUTPUT || (qs.has_tied_embeddings && category == tensor_category::TOKEN_EMBD)) {
        if (qs.params->output_tensor_type < GGML_TYPE_COUNT) {
            new_type = qs.params->output_tensor_type;
        } else {
            const int64_t nx = tensor->ne[0];
            const int64_t qk_k = ggml_blck_size(new_type);

            if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (arch == LLM_ARCH_FALCON || nx % qk_k != 0) {
                new_type = GGML_TYPE_Q8_0;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ2_S  || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M   ||
                     ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q5_K;
            }
            else if (new_type != GGML_TYPE_Q8_0) {
                new_type = GGML_TYPE_Q6_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
        // MoE   tensors -> MXFP4
        // other tensors -> Q8_0
        if (tensor->ne[2] > 1) {
            new_type = GGML_TYPE_MXFP4;
        } else {
            new_type = GGML_TYPE_Q8_0;
        }
    } else if (category == tensor_category::TOKEN_EMBD) {
        if (qs.params->token_embedding_type < GGML_TYPE_COUNT) {
            new_type = qs.params->token_embedding_type;
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS ||
                ftype == LLAMA_FTYPE_MOSTLY_IQ1_S   || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
                new_type = GGML_TYPE_Q2_K;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
                new_type = GGML_TYPE_IQ3_S;
            }
            else if (ftype == LLAMA_FTYPE_MOSTLY_TQ1_0 || ftype == LLAMA_FTYPE_MOSTLY_TQ2_0) {
                new_type = GGML_TYPE_Q4_K;
            }
        }
    } else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_XXS || ftype == LLAMA_FTYPE_MOSTLY_IQ2_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ1_S ||
               ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M    || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) {
        if (category_is_attn_v(category)) {
            if (qs.model.hparams.n_gqa() >= 4 || qs.model.hparams.n_expert >= 4) new_type = GGML_TYPE_Q4_K;
            else new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            ++qs.i_attention_wv;
        }
        else if (qs.model.hparams.n_expert == 8 && category == tensor_category::ATTENTION_K) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (category == tensor_category::FFN_DOWN) {
            if (qs.i_ffn_down < qs.n_ffn_down/8) {
                new_type = ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M ? GGML_TYPE_IQ3_S : GGML_TYPE_Q2_K;
            }
            ++qs.i_ffn_down;
        }
        else if (category == tensor_category::ATTENTION_OUTPUT) {
            if (qs.model.hparams.n_expert == 8) {
                new_type = GGML_TYPE_Q5_K;
            } else {
                if (ftype == LLAMA_FTYPE_MOSTLY_IQ1_S || ftype == LLAMA_FTYPE_MOSTLY_IQ1_M) new_type = GGML_TYPE_IQ2_XXS;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ2_S || ftype == LLAMA_FTYPE_MOSTLY_IQ2_M) new_type = GGML_TYPE_IQ3_S;
            }
        }
    } else if (category_is_attn_v(category)) {
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = qs.model.hparams.n_gqa() >= 4 ? GGML_TYPE_Q4_K : !qs.has_imatrix ? GGML_TYPE_IQ3_S : GGML_TYPE_IQ3_XXS;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = qs.i_attention_wv < 2 ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q5_K;
        else if ((ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && qs.model.hparams.n_gqa() >= 4) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) &&
                use_more_bits(qs.i_attention_wv, qs.n_attention_wv)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && qs.i_attention_wv < 4) new_type = GGML_TYPE_Q5_K;
        if (qs.model.type == LLM_TYPE_70B) {
            // In the 70B model we have 8 heads sharing the same attn_v weights. As a result, the attn_v.weight tensor is
            // 8x smaller compared to attn_q.weight. Hence, we can get a nice boost in quantization accuracy with
            // nearly negligible increase in model size by quantizing this tensor with more bits:
            if (new_type == GGML_TYPE_Q3_K || new_type == GGML_TYPE_Q4_K) new_type = GGML_TYPE_Q5_K;
        }
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        ++qs.i_attention_wv;
    } else if (category == tensor_category::ATTENTION_K) {
        if (qs.model.hparams.n_expert == 8) {
            // for the 8-expert model, bumping this to Q8_0 trades just ~128MB
            // TODO: explore better strategies
            new_type = GGML_TYPE_Q8_0;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::ATTENTION_Q) {
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) {
            new_type = GGML_TYPE_IQ2_S;
        }
    } else if (category == tensor_category::FFN_DOWN) {
        auto info = layer_info(qs.i_ffn_down, qs.n_ffn_down, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K) new_type = GGML_TYPE_Q3_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S) {
            if (i_layer < n_layer/8) new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS && !qs.has_imatrix) {
            new_type = i_layer < n_layer/8 ? GGML_TYPE_Q4_K : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M) {
            new_type = i_layer < n_layer/16 ? GGML_TYPE_Q5_K
                     : arch != LLM_ARCH_FALCON || use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q4_K
                     : GGML_TYPE_Q3_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M && (i_layer < n_layer/8 ||
                    (qs.model.hparams.n_expert == 8 && use_more_bits(i_layer, n_layer)))) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) {
            new_type = arch == LLM_ARCH_FALCON ? GGML_TYPE_Q4_K : GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) {
            if (arch == LLM_ARCH_FALCON) {
                new_type = i_layer < n_layer/16 ? GGML_TYPE_Q6_K :
                           use_more_bits(i_layer, n_layer) ? GGML_TYPE_Q5_K : GGML_TYPE_Q4_K;
            } else {
                if (use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
            }
        }
        else if (i_layer < n_layer/8 && (ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) && !qs.has_imatrix) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M && use_more_bits(i_layer, n_layer)) new_type = GGML_TYPE_Q6_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S && arch != LLM_ARCH_FALCON && i_layer < n_layer/8) {
            new_type = GGML_TYPE_Q5_K;
        }
        else if ((ftype == LLAMA_FTYPE_MOSTLY_Q4_0 || ftype == LLAMA_FTYPE_MOSTLY_Q5_0)
                && qs.has_imatrix && i_layer < n_layer/8) {
            // Guard against craziness in the first few ffn_down layers that can happen even with imatrix for Q4_0/Q5_0.
            // We only do it when an imatrix is provided because a) we want to make sure that one can always get the
            // same quantization as before imatrix stuff, and b) Q4_1/Q5_1 do go crazy on ffn_down without an imatrix.
            new_type = ftype == LLAMA_FTYPE_MOSTLY_Q4_0 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q5_1;
        }
        ++qs.i_ffn_down;
    } else if (category == tensor_category::ATTENTION_OUTPUT) {
        if (arch != LLM_ARCH_FALCON) {
            if (qs.model.hparams.n_expert == 8) {
                if (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS || ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q3_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_NL  ||
                    ftype == LLAMA_FTYPE_MOSTLY_Q4_K_S || ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ3_S  ||
                    ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  || ftype == LLAMA_FTYPE_MOSTLY_IQ4_XS) {
                    new_type = GGML_TYPE_Q5_K;
                }
            } else {
                if      (ftype == LLAMA_FTYPE_MOSTLY_Q2_K   ) new_type = GGML_TYPE_Q3_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XXS) new_type = GGML_TYPE_IQ3_S;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M ) new_type = GGML_TYPE_Q4_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L ) new_type = GGML_TYPE_Q5_K;
                else if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_M  ) new_type = GGML_TYPE_Q4_K;
            }
        } else {
            if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L) new_type = GGML_TYPE_Q4_K;
        }
    }
    else if (category == tensor_category::ATTENTION_QKV) {
        if (ftype == LLAMA_FTYPE_MOSTLY_Q3_K_M || ftype == LLAMA_FTYPE_MOSTLY_Q3_K_L || ftype == LLAMA_FTYPE_MOSTLY_IQ3_M) {
            new_type = GGML_TYPE_Q4_K;
        }
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q4_K_M) new_type = GGML_TYPE_Q5_K;
        else if (ftype == LLAMA_FTYPE_MOSTLY_Q5_K_M) new_type = GGML_TYPE_Q6_K;
    }
    else if (category == tensor_category::FFN_GATE) {
        auto info = layer_info(qs.i_ffn_gate, qs.n_ffn_gate, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_gate;
    }
    else if (category == tensor_category::FFN_UP) {
        auto info = layer_info(qs.i_ffn_up, qs.n_ffn_up, name.c_str());
        int i_layer = info.first, n_layer = info.second;
        if (ftype == LLAMA_FTYPE_MOSTLY_IQ3_XS && (i_layer >= n_layer/8 && i_layer < 7*n_layer/8)) {
            new_type = GGML_TYPE_IQ3_XXS;
        }
        ++qs.i_ffn_up;
    }

    return new_type;
}

// outer wrapper: determine the ggml_type that this tensor should be quantized to
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, const tensor_metadata & tm) {
    if (!tensor_allows_quantization(params, qs.model.arch, tensor)) {
        return tensor->type;
    }
    if (params->token_embedding_type < GGML_TYPE_COUNT && tm.category == tensor_category::TOKEN_EMBD) {
        return params->token_embedding_type;
    }
    if (params->output_tensor_type < GGML_TYPE_COUNT && tm.category == tensor_category::OUTPUT) {
        return params->output_tensor_type;
    }

    ggml_type new_type = default_type;

    // get more optimal quantization type based on the tensor shape, layer, etc.
    if (!params->pure && ggml_is_quantized(default_type)) {
        // if the user provided tensor types - use those
        bool manual = false;
        if (!qs.tensor_type_patterns.empty()) {
            const std::string tensor_name(tensor->name);
            for (const auto & [pattern, qtype] : qs.tensor_type_patterns) {
                if (std::regex_search(tensor_name, pattern)) {
                    if (qtype != new_type) {
                        LLAMA_LOG_WARN("%s: %-36s - applying manual override: %s -> %s\n",
                                       __func__, tensor_name.c_str(), ggml_type_name(new_type), ggml_type_name(qtype));
                        new_type = qtype;
                    }
                    manual = true;
                    break;
                }
            }
        }

        // if not manual - use the standard logic for choosing the quantization type based on the selected mixture
        if (!manual) {
            if (params->rd_guided && tm.rd_type != GGML_TYPE_COUNT) {
                new_type = tm.rd_type;
            } else if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED) {
                new_type = spqr_guided_type_for_bucket(params, tm.category, default_type, tm.sensitivity);
            } else if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) {
                new_type = spqr_layer_delta_type_for_bucket(
                        params, tm.category, default_type, tm.sensitivity, tm.layer_similarity, tm.anchor_reason != ANCHOR_REASON_NONE);
            } else {
                new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
            }
        }

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);

        if ((params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
                    params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) && !manual) {
            const int64_t ncols = tensor->ne[0];
            const bool can_compare =
                ncols % ggml_blck_size(default_type) == 0 &&
                ncols % ggml_blck_size(new_type) == 0;
            if (can_compare && ggml_row_size(new_type, ncols) > ggml_row_size(default_type, ncols)) {
                ++qs.n_spqr_promoted;
            } else if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA &&
                    can_compare && ggml_row_size(new_type, ncols) < ggml_row_size(default_type, ncols)) {
                ++qs.n_delta_demoted;
            }
        }
    }

    return new_type;
}

//
// quantization implementation
//

static size_t llama_tensor_quantize_impl(enum ggml_type new_type, const float * f32_data, void * new_data, const int64_t chunk_size, int64_t nrows, int64_t n_per_row, const float * imatrix, std::vector<std::thread> & workers, const int nthread) {
    if (nthread < 2) {
        // single-thread
        size_t new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nrows, n_per_row, imatrix);
        if (!ggml_validate_row_data(new_type, new_data, new_size)) {
            throw std::runtime_error("quantized data validation failed");
        }
        return new_size;
    }

    std::mutex mutex;
    int64_t counter = 0;
    size_t new_size = 0;
    bool valid = true;
    auto compute = [&mutex, &counter, &new_size, &valid, new_type, f32_data, new_data, chunk_size,
            nrows, n_per_row, imatrix]() {
        const int64_t nrows_per_chunk = chunk_size / n_per_row;
        size_t local_size = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            int64_t first_row = counter; counter += nrows_per_chunk;
            if (first_row >= nrows) {
                if (local_size > 0) {
                    new_size += local_size;
                }
                break;
            }
            lock.unlock();
            const int64_t this_nrow = std::min(nrows - first_row, nrows_per_chunk);
            size_t this_size = ggml_quantize_chunk(new_type, f32_data, new_data, first_row * n_per_row, this_nrow, n_per_row, imatrix);
            local_size += this_size;

            // validate the quantized data
            const size_t row_size  = ggml_row_size(new_type, n_per_row);
            void * this_data = (char *) new_data + first_row * row_size;
            if (!ggml_validate_row_data(new_type, this_data, this_size)) {
                std::unique_lock<std::mutex> lock(mutex);
                valid = false;
                break;
            }
        }
    };
    for (int it = 0; it < nthread - 1; ++it) {
        workers.emplace_back(compute);
    }
    compute();
    for (auto & w : workers) { w.join(); }
    workers.clear();
    if (!valid) {
        throw std::runtime_error("quantized data validation failed");
    }
    return new_size;
}

//
// imatrix requirement check
//

static bool tensor_requires_imatrix(const char * tensor_name, const ggml_type dst_type, const llama_ftype ftype) {
    if (tensor_name_match_token_embd(tensor_name) || tensor_name_match_output_weight(tensor_name)) {
        return false;
    }
    switch (dst_type) {
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ1_S:
            return true;
        case GGML_TYPE_Q2_K:
            // as a general rule, the k-type quantizations don't require imatrix data.
            // the only exception is Q2_K tensors that are part of a Q2_K_S file.
            return ftype == LLAMA_FTYPE_MOSTLY_Q2_K_S;
        default:
            return false;
    }
}

//
// given a file type, get the default tensor type
//

ggml_type llama_ftype_get_default_type(llama_ftype ftype) {
    switch (ftype) {
        case LLAMA_FTYPE_MOSTLY_Q4_0: return GGML_TYPE_Q4_0;
        case LLAMA_FTYPE_MOSTLY_Q4_1: return GGML_TYPE_Q4_1;
        case LLAMA_FTYPE_MOSTLY_Q5_0: return GGML_TYPE_Q5_0;
        case LLAMA_FTYPE_MOSTLY_Q5_1: return GGML_TYPE_Q5_1;
        case LLAMA_FTYPE_MOSTLY_Q8_0: return GGML_TYPE_Q8_0;
        case LLAMA_FTYPE_MOSTLY_F16:  return GGML_TYPE_F16;
        case LLAMA_FTYPE_MOSTLY_BF16: return GGML_TYPE_BF16;
        case LLAMA_FTYPE_ALL_F32:     return GGML_TYPE_F32;
        case LLAMA_FTYPE_MOSTLY_Q1_0: return GGML_TYPE_Q1_0;

        case LLAMA_FTYPE_MOSTLY_MXFP4_MOE: return GGML_TYPE_MXFP4;

        // K-quants
        case LLAMA_FTYPE_MOSTLY_Q2_K_S:
        case LLAMA_FTYPE_MOSTLY_Q2_K:    return GGML_TYPE_Q2_K;
        case LLAMA_FTYPE_MOSTLY_IQ3_XS:  return GGML_TYPE_IQ3_S;
        case LLAMA_FTYPE_MOSTLY_Q3_K_S:
        case LLAMA_FTYPE_MOSTLY_Q3_K_M:
        case LLAMA_FTYPE_MOSTLY_Q3_K_L:  return GGML_TYPE_Q3_K;
        case LLAMA_FTYPE_MOSTLY_Q4_K_S:
        case LLAMA_FTYPE_MOSTLY_Q4_K_M:  return GGML_TYPE_Q4_K;
        case LLAMA_FTYPE_MOSTLY_Q5_K_S:
        case LLAMA_FTYPE_MOSTLY_Q5_K_M:  return GGML_TYPE_Q5_K;
        case LLAMA_FTYPE_MOSTLY_Q6_K:    return GGML_TYPE_Q6_K;
        case LLAMA_FTYPE_MOSTLY_TQ1_0:   return GGML_TYPE_TQ1_0;
        case LLAMA_FTYPE_MOSTLY_TQ2_0:   return GGML_TYPE_TQ2_0;
        case LLAMA_FTYPE_MOSTLY_IQ2_XXS: return GGML_TYPE_IQ2_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ2_XS:  return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_S:   return GGML_TYPE_IQ2_XS;
        case LLAMA_FTYPE_MOSTLY_IQ2_M:   return GGML_TYPE_IQ2_S;
        case LLAMA_FTYPE_MOSTLY_IQ3_XXS: return GGML_TYPE_IQ3_XXS;
        case LLAMA_FTYPE_MOSTLY_IQ1_S:   return GGML_TYPE_IQ1_S;
        case LLAMA_FTYPE_MOSTLY_IQ1_M:   return GGML_TYPE_IQ1_M;
        case LLAMA_FTYPE_MOSTLY_IQ4_NL:  return GGML_TYPE_IQ4_NL;
        case LLAMA_FTYPE_MOSTLY_IQ4_XS:  return GGML_TYPE_IQ4_XS;
        case LLAMA_FTYPE_MOSTLY_IQ3_S:
        case LLAMA_FTYPE_MOSTLY_IQ3_M:   return GGML_TYPE_IQ3_S;

        default: return GGML_TYPE_COUNT;
    }
}


static void init_quantize_state_counters(quantize_state_impl & qs, std::vector<tensor_metadata> & metadata) {
    for (auto & tm : metadata) {
        tensor_category cat = tensor_get_category(tm.name);
        tm.category = cat;
        tm.sensitivity = sensitivity_bucket::UNKNOWN;
        tm.sensitivity_from = sensitivity_source::NONE;
        tm.sensitivity_score = 0.0f;
        tm.block_low = 0;
        tm.block_medium = 0;
        tm.block_high = 0;
        tm.block_count = 0;
        tm.layer = -1;
        tm.rel_delta_norm = 0.0f;
        tm.cosine_similarity = 0.0f;
        tm.layer_similarity = layer_similarity_bucket::UNKNOWN;
        tm.anchor_reason = ANCHOR_REASON_NONE;
        tm.rd_type = GGML_TYPE_COUNT;
        tm.rd_cost = 0.0f;
        tm.rd_distortion = 0.0f;
        tm.rd_bpw = 0.0f;

        if (category_is_attn_v(cat)) {
            ++qs.n_attention_wv;
        }

        if (cat == tensor_category::OUTPUT) {
            qs.has_tied_embeddings = false;
        }
    }
    qs.n_ffn_down = qs.n_ffn_gate = qs.n_ffn_up = (int)qs.model.hparams.n_layer;
}

static float percentile(std::vector<float> values, float p) {
    GGML_ASSERT(!values.empty());
    std::sort(values.begin(), values.end());
    const float pos = p * (values.size() - 1);
    const size_t lo = (size_t) std::floor(pos);
    const size_t hi = (size_t) std::ceil(pos);
    if (lo == hi) {
        return values[lo];
    }
    const float t = pos - lo;
    return values[lo] * (1.0f - t) + values[hi] * t;
}

static float imatrix_tensor_score(const std::vector<float> & data) {
    if (data.empty()) {
        return 0.0f;
    }

    double sum = 0.0;
    size_t count = 0;
    for (float v : data) {
        if (std::isfinite(v)) {
            sum += std::abs(v);
            ++count;
        }
    }

    return count == 0 ? 0.0f : (float) (sum / count);
}

static sensitivity_bucket bucket_from_score(float score, float medium_threshold, float high_threshold) {
    if (score >= high_threshold) {
        return sensitivity_bucket::HIGH;
    }
    if (score >= medium_threshold) {
        return sensitivity_bucket::MEDIUM;
    }
    return sensitivity_bucket::LOW;
}

static void init_spqr_guided_block_scoring(
        quantize_state_impl & qs,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int32_t block_size,
        bool apply_scoring) {
    if (block_size <= 0) {
        block_size = 256;
    }

    std::vector<float> block_scores;
    for (auto & tm : metadata) {
        if (!tm.allows_quantization || !imatrix_data) {
            continue;
        }

        auto it = imatrix_data->find(tm.remapped_imatrix_name);
        if (it == imatrix_data->end() || it->second.empty()) {
            continue;
        }

        const auto & data = it->second;
        for (size_t off = 0; off < data.size(); off += (size_t) block_size) {
            const size_t end = std::min(data.size(), off + (size_t) block_size);
            double sum = 0.0;
            size_t count = 0;
            for (size_t i = off; i < end; ++i) {
                if (std::isfinite(data[i])) {
                    sum += std::abs(data[i]);
                    ++count;
                }
            }
            if (count > 0) {
                block_scores.push_back((float) (sum / count));
            }
        }
    }

    const bool use_imatrix_blocks = block_scores.size() >= 3;
    const float medium_threshold = use_imatrix_blocks ? percentile(block_scores, 0.40f) : 0.0f;
    const float high_threshold   = use_imatrix_blocks ? percentile(block_scores, 0.75f) : 0.0f;

    for (auto & tm : metadata) {
        if (!tm.allows_quantization) {
            continue;
        }

        if (use_imatrix_blocks && imatrix_data) {
            auto it = imatrix_data->find(tm.remapped_imatrix_name);
            if (it != imatrix_data->end() && !it->second.empty()) {
                const auto & data = it->second;
                for (size_t off = 0; off < data.size(); off += (size_t) block_size) {
                    const size_t end = std::min(data.size(), off + (size_t) block_size);
                    double sum = 0.0;
                    size_t count = 0;
                    for (size_t i = off; i < end; ++i) {
                        if (std::isfinite(data[i])) {
                            sum += std::abs(data[i]);
                            ++count;
                        }
                    }
                    if (count == 0) {
                        continue;
                    }

                    const sensitivity_bucket bucket = bucket_from_score((float) (sum / count), medium_threshold, high_threshold);
                    switch (bucket) {
                        case sensitivity_bucket::LOW:    ++tm.block_low;    break;
                        case sensitivity_bucket::MEDIUM: ++tm.block_medium; break;
                        case sensitivity_bucket::HIGH:   ++tm.block_high;   break;
                        case sensitivity_bucket::UNKNOWN: break;
                    }
                    ++tm.block_count;
                }
                continue;
            }
        }

        tm.block_count = 1;
        switch (tm.sensitivity) {
            case sensitivity_bucket::LOW:    tm.block_low    = 1; break;
            case sensitivity_bucket::MEDIUM: tm.block_medium = 1; break;
            case sensitivity_bucket::HIGH:   tm.block_high   = 1; break;
            case sensitivity_bucket::UNKNOWN: break;
        }
    }

    if (apply_scoring && use_imatrix_blocks) {
        qs.n_spqr_low = qs.n_spqr_medium = qs.n_spqr_high = 0;
        qs.n_spqr_block_scored = 0;

        for (auto & tm : metadata) {
            if (!tm.allows_quantization) {
                continue;
            }

            if (tm.block_count > 1) {
                const float high_fraction = (float) tm.block_high / tm.block_count;
                const float elevated_fraction = (float) (tm.block_high + tm.block_medium) / tm.block_count;

                // A small sensitive region should protect the tensor, but a single
                // noisy block should not promote the entire tensor.
                if (high_fraction >= 0.35f) {
                    tm.sensitivity = sensitivity_bucket::HIGH;
                } else if (high_fraction >= 0.15f || elevated_fraction >= 0.50f) {
                    tm.sensitivity = sensitivity_bucket::MEDIUM;
                } else {
                    tm.sensitivity = sensitivity_bucket::LOW;
                }
                tm.sensitivity_score = 2.0f * high_fraction + elevated_fraction;
                tm.sensitivity_from = sensitivity_source::IMATRIX_BLOCKS;
                ++qs.n_spqr_block_scored;
            }

            switch (tm.sensitivity) {
                case sensitivity_bucket::LOW:    ++qs.n_spqr_low;    break;
                case sensitivity_bucket::MEDIUM: ++qs.n_spqr_medium; break;
                case sensitivity_bucket::HIGH:   ++qs.n_spqr_high;   break;
                case sensitivity_bucket::UNKNOWN: break;
            }
        }
    }

    LLAMA_LOG_INFO("%s: SPQR-guided block sensitivity %s enabled (block_size=%d, %s)\n",
            __func__,
            apply_scoring ? "scoring" : "report",
            block_size,
            use_imatrix_blocks ? "imatrix block percentiles" : "tensor-bucket fallback");
    if (apply_scoring && !use_imatrix_blocks) {
        LLAMA_LOG_WARN("%s: block scoring requested but insufficient imatrix block data; keeping tensor-level sensitivity\n", __func__);
    }
}

static void init_spqr_guided_sensitivity(
        quantize_state_impl & qs,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data) {
    std::vector<float> imatrix_scores;
    imatrix_scores.reserve(metadata.size());

    for (auto & tm : metadata) {
        if (!tm.allows_quantization || !imatrix_data) {
            continue;
        }

        auto it = imatrix_data->find(tm.remapped_imatrix_name);
        if (it == imatrix_data->end()) {
            continue;
        }

        tm.sensitivity_score = imatrix_tensor_score(it->second);
        tm.sensitivity_from  = sensitivity_source::IMATRIX;
        imatrix_scores.push_back(tm.sensitivity_score);
    }

    const bool use_imatrix_percentiles = imatrix_scores.size() >= 3;
    const float medium_threshold = use_imatrix_percentiles ? percentile(imatrix_scores, 0.40f) : 0.0f;
    const float high_threshold   = use_imatrix_percentiles ? percentile(imatrix_scores, 0.75f) : 0.0f;

    for (auto & tm : metadata) {
        if (!tm.allows_quantization) {
            continue;
        }

        if (tm.sensitivity_from == sensitivity_source::IMATRIX && use_imatrix_percentiles) {
            tm.sensitivity = bucket_from_score(tm.sensitivity_score, medium_threshold, high_threshold);
        } else {
            tm.sensitivity = spqr_guided_heuristic_bucket(tm.category);
            tm.sensitivity_from = sensitivity_source::HEURISTIC;
        }

        switch (tm.sensitivity) {
            case sensitivity_bucket::LOW:    ++qs.n_spqr_low;    break;
            case sensitivity_bucket::MEDIUM: ++qs.n_spqr_medium; break;
            case sensitivity_bucket::HIGH:   ++qs.n_spqr_high;   break;
            case sensitivity_bucket::UNKNOWN: break;
        }
    }

    LLAMA_LOG_INFO("%s: SPQR-guided mixed quantization enabled (%s sensitivity for %d tensor(s), heuristic fallback for the rest)\n",
            __func__,
            use_imatrix_percentiles ? "imatrix percentile" : "heuristic",
            (int) imatrix_scores.size());
}

// SpQR-inspired sensitivity and layer similarity are useful steering signals, but the output
// remains a normal GGUF tensor using one existing ggml_type. This sampled rate-distortion pass
// provides an extension point for richer block-level policies without adding a runtime format.
static void init_rate_distortion_analysis(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        ggml_type default_type,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    const int max_sample_rows = std::max(1, params->rd_sample_rows);

    std::vector<ggml_type> candidates = {
        default_type,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
    };
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;

        // Keep explicit safety floors and adaptive anchors under the existing mixed policy.
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                tm.anchor_reason != ANCHOR_REASON_NONE) {
            continue;
        }

        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (ncols <= 0 || nrows <= 0) {
            continue;
        }

        const float * imatrix = nullptr;
        if (imatrix_data) {
            auto it = imatrix_data->find(tm.remapped_imatrix_name);
            if (it != imatrix_data->end() && it->second.size() == (size_t) ncols * tensor->ne[2]) {
                imatrix = it->second.data();
            }
        }

        load_tensor_as_f32(ml, tensor, read_data, f32_conv, data, workers, nthread);

        float distortion_weight = 1.0f;
        if (tm.sensitivity == sensitivity_bucket::HIGH) {
            distortion_weight = 2.0f;
        } else if (tm.sensitivity == sensitivity_bucket::MEDIUM) {
            distortion_weight = 1.35f;
        }
        if (tm.layer_similarity == layer_similarity_bucket::LOW) {
            distortion_weight *= 1.20f;
        } else if (tm.layer_similarity == layer_similarity_bucket::HIGH) {
            distortion_weight *= 0.85f;
        }

        const int sample_rows = (int) std::min<int64_t>(nrows, max_sample_rows);
        float best_cost = std::numeric_limits<float>::infinity();
        ggml_type best_type = GGML_TYPE_COUNT;
        float best_distortion = 0.0f;
        float best_bpw = 0.0f;

        for (ggml_type candidate : candidates) {
            if (!ggml_is_quantized(candidate) ||
                    ncols % ggml_blck_size(candidate) != 0 ||
                    (ggml_quantize_requires_imatrix(candidate) && !imatrix)) {
                continue;
            }

            const ggml_type_traits * traits = ggml_get_type_traits(candidate);
            if (!traits || !traits->to_float) {
                continue;
            }

            std::vector<no_init<uint8_t>> quantized(ggml_row_size(candidate, ncols));
            std::vector<no_init<float>> reconstructed(ncols);
            double error_sum = 0.0;
            double signal_sum = 0.0;

            for (int sample = 0; sample < sample_rows; ++sample) {
                const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
                const float * src = data.data() + row * ncols;
                const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
                const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;

                ggml_quantize_chunk(candidate, src, quantized.data(), 0, 1, ncols, row_imatrix);
                traits->to_float(quantized.data(), reconstructed.data(), ncols);

                for (int64_t col = 0; col < ncols; ++col) {
                    const double weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0;
                    const double delta = (double) src[col] - reconstructed[col];
                    error_sum += weight * delta * delta;
                    signal_sum += weight * src[col] * src[col];
                }
            }

            const float distortion = (float) (error_sum / (signal_sum + 1e-20));
            const float bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
            const float cost = distortion_weight * distortion + params->rd_lambda * bpw;

            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: rd-candidate tensor=%-36s type=%-7s samples=%3d distortion=%10.6g weight=%5.2f bpw=%6.3f cost=%10.6g\n",
                        __func__, tm.name.c_str(), ggml_type_name(candidate), sample_rows,
                        distortion, distortion_weight, bpw, cost);
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_type = candidate;
                best_distortion = distortion;
                best_bpw = bpw;
            }
        }

        if (best_type != GGML_TYPE_COUNT) {
            tm.rd_type = best_type;
            tm.rd_cost = best_cost;
            tm.rd_distortion = best_distortion;
            tm.rd_bpw = best_bpw;
            ++qs.n_rd_selected;

            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: rd-selected  tensor=%-36s type=%-7s distortion=%10.6g bpw=%6.3f cost=%10.6g\n",
                        __func__, tm.name.c_str(), ggml_type_name(best_type),
                        best_distortion, best_bpw, best_cost);
            }
        }
    }

    LLAMA_LOG_INFO("%s: sampled rate-distortion analysis selected existing quant types for %d tensor(s) (lambda=%.6g, max_sample_rows=%d)\n",
            __func__, qs.n_rd_selected, params->rd_lambda, max_sample_rows);
}

//
// main quantization driver
//

static void llama_model_quantize_impl(const std::string & fname_inp, const std::string & fname_out, const llama_model_quantize_params * params) {
    llama_ftype ftype = params->ftype;

    int nthread = params->nthread;

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);
    if (default_type == GGML_TYPE_COUNT) {
        throw std::runtime_error(format("invalid output file type %d\n", ftype));
    }

    // mmap consistently increases speed on Linux, and also increases speed on Windows with
    // hot cache. It may cause a slowdown on macOS, possibly related to free memory.
#if defined(__linux__) || defined(_WIN32)
    constexpr bool use_mmap = true;
#else
    constexpr bool use_mmap = false;
#endif

    const llama_model_kv_override * kv_overrides = params->kv_overrides;
    std::vector<std::string> splits = {};
    llama_model_loader ml(/*metadata*/ nullptr, /*set_tensor_data*/ nullptr, /*set_tensor_data_ud*/ nullptr,
        fname_inp, splits, /*file*/ nullptr, use_mmap, /*use_direct_io*/ false, /*check_tensors*/ true, /*no_alloc*/ false, kv_overrides, nullptr);
    ml.init_mappings(false); // no prefetching

    auto mparams = llama_model_default_params();
    std::unique_ptr<llama_model> model_ptr(llama_model_create(ml, mparams));

    auto * model = dynamic_cast<llama_model_base *>(model_ptr.get());
    if (model == nullptr) {
        GGML_ABORT("fatal error: model does not implement llama_model_base");
    }

    model->load_hparams(ml);
    model->load_stats  (ml);

    quantize_state_impl qs(*model, params);

    if (params->only_copy) {
        ftype = ml.ftype;
    }
    std::unordered_map<std::string, std::vector<float>> i_data;
    const std::unordered_map<std::string, std::vector<float>> * imatrix_data = nullptr;
    if (params->imatrix) {
        for (const llama_model_imatrix_data * p = params->imatrix; p->name != nullptr; p++) {
            i_data.emplace(p->name, std::vector<float>(p->data, p->data + p->size));
        }
        imatrix_data = & i_data;
        if (imatrix_data) {
            LLAMA_LOG_INFO("\n%s: have importance matrix data with %d entries\n",
                           __func__, (int)imatrix_data->size());
            qs.has_imatrix = true;
            // check imatrix for nans or infs
            for (const auto & kv : *imatrix_data) {
                for (float f : kv.second) {
                    if (!std::isfinite(f)) {
                        throw std::runtime_error(format("imatrix contains non-finite value %f\n", f));
                    }
                }
            }
        }
    }

    const size_t align = GGUF_DEFAULT_ALIGNMENT;
    gguf_context_ptr ctx_out { gguf_init_empty() };

    std::vector<int> prune_list = {};
    if (params->prune_layers) {
        for (const int32_t * p = params->prune_layers; * p != -1; p++) {
            prune_list.push_back(* p);
        }
    }

    // copy the KV pairs from the input file
    gguf_set_kv     (ctx_out.get(), ml.metadata);
    gguf_set_val_u32(ctx_out.get(), "general.quantization_version", GGML_QNT_VERSION); // TODO: use LLM_KV
    gguf_set_val_u32(ctx_out.get(), "general.file_type", ftype); // TODO: use LLM_KV

    // Remove split metadata
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str());
    gguf_remove_key(ctx_out.get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str());

    if (params->kv_overrides) {
        for (const llama_model_kv_override * o = params->kv_overrides; o->key[0] != 0; ++o) {
            if (o->tag == LLAMA_KV_OVERRIDE_TYPE_FLOAT) {
                gguf_set_val_f32(ctx_out.get(), o->key, o->val_f64);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_INT) {
                // Setting type to UINT32. See https://github.com/ggml-org/llama.cpp/pull/14182 for context
                gguf_set_val_u32(ctx_out.get(), o->key, (uint32_t)std::abs(o->val_i64));
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_BOOL) {
                gguf_set_val_bool(ctx_out.get(), o->key, o->val_bool);
            } else if (o->tag == LLAMA_KV_OVERRIDE_TYPE_STR) {
                gguf_set_val_str(ctx_out.get(), o->key, o->val_str);
            } else {
                LLAMA_LOG_WARN("%s: unknown KV override type for key %s\n", __func__, o->key);
            }
        }
    }

    std::map<int, std::string> mapped;
    int blk_id = 0;

    // make a list of weights
    std::vector<const llama_model_loader::llama_tensor_weight *> tensors;
    tensors.reserve(ml.weights_map.size());
    for (const auto & it : ml.weights_map) {
        const std::string remapped_name(remap_layer(it.first, prune_list, mapped, blk_id));
        if (remapped_name.empty()) {
            LLAMA_LOG_DEBUG("%s: pruning tensor %s\n", __func__, it.first.c_str());
            continue;
        }

        if (remapped_name != it.first) {
            ggml_set_name(it.second.tensor, remapped_name.c_str());
            LLAMA_LOG_DEBUG("%s: tensor %s remapped to %s\n", __func__, it.first.c_str(), ggml_get_name(it.second.tensor));
        }
        tensors.push_back(&it.second);
    }
    if (!prune_list.empty()) {
        gguf_set_val_u32(ctx_out.get(), ml.llm_kv(LLM_KV_BLOCK_COUNT).c_str(), blk_id);
    }

    // keep_split requires that the weights are sorted by split index
    if (params->keep_split) {
        std::sort(tensors.begin(), tensors.end(), [](const llama_model_loader::llama_tensor_weight * a, const llama_model_loader::llama_tensor_weight * b) {
            if (a->idx == b->idx) {
                return a->offs < b->offs;
            }
            return a->idx < b->idx;
        });
    }

    // compute tensor metadata once and cache it
    std::vector<tensor_metadata> metadata(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        metadata[i].name = ggml_get_name(tensors[i]->tensor);
    }

    // initialize quantization state counters and metadata categories
    init_quantize_state_counters(qs, metadata);

    for (size_t i = 0; i < tensors.size(); ++i) {
        const struct ggml_tensor * tensor = tensors[i]->tensor;
        metadata[i].allows_quantization = tensor_allows_quantization(params, model->arch, tensor);
        metadata[i].remapped_imatrix_name = params->imatrix ? remap_imatrix(tensor->name, mapped) : metadata[i].name;
    }

    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA ||
            params->print_layer_delta_report ||
            params->rd_guided) {
        init_spqr_guided_sensitivity(qs, metadata, imatrix_data);
        if (params->spqr_block_report || params->spqr_block_scoring) {
            init_spqr_guided_block_scoring(qs, metadata, imatrix_data, params->spqr_block_size, params->spqr_block_scoring);
        }
    }
    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA || params->print_layer_delta_report) {
        init_layer_delta_analysis(qs, ml, tensors, metadata, nthread, params->print_layer_delta_report);
    }
    if (params->rd_guided) {
        init_rate_distortion_analysis(qs, ml, tensors, metadata, imatrix_data, default_type, nthread);
    }

    int idx = 0;
    uint16_t n_split = 1;

    // Assume split index is continuous
    if (params->keep_split) {
        for (const auto * it : tensors) {
            n_split = std::max(uint16_t(it->idx + 1), n_split);
        }
    }
    std::vector<gguf_context_ptr> ctx_outs(n_split);
    ctx_outs[0] = std::move(ctx_out);

    // flag for --dry-run
    bool will_require_imatrix = false;

    //
    // preliminary iteration over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * it = tensors[i];
        const struct ggml_tensor * tensor = it->tensor;

        uint16_t i_split = params->keep_split ? it->idx : 0;
        if (!ctx_outs[i_split]) {
            ctx_outs[i_split].reset(gguf_init_empty());
        }
        gguf_add_tensor(ctx_outs[i_split].get(), tensor);

        if (metadata[i].allows_quantization) {
            metadata[i].target_type = llama_tensor_get_type(qs, params, tensor, default_type, metadata[i]);
        } else {
            metadata[i].target_type = tensor->type;
        }

        metadata[i].requires_imatrix = tensor_requires_imatrix(tensor->name, metadata[i].target_type, ftype);

        if (!params->imatrix && metadata[i].allows_quantization && metadata[i].requires_imatrix) {
            if (params->dry_run) {
                will_require_imatrix = true;
            } else {
                LLAMA_LOG_ERROR("\n============================================================================\n"
                                " ERROR: this quantization requires an importance matrix!\n"
                                "        - offending tensor: %s\n"
                                "        - target type: %s\n"
                                "============================================================================\n\n",
                                metadata[i].name.c_str(), ggml_type_name(metadata[i].target_type));
                throw std::runtime_error("this quantization requires an imatrix!");
            }
        }
    }

    // Set split info if needed
    if (n_split > 1) {
        for (size_t i = 0; i < ctx_outs.size(); ++i) {
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_NO).c_str(), i);
            gguf_set_val_u16(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_COUNT).c_str(), n_split);
            gguf_set_val_i32(ctx_outs[i].get(), ml.llm_kv(LLM_KV_SPLIT_TENSORS_COUNT).c_str(), (int32_t)tensors.size());
        }
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<uint8_t>> work;
    std::vector<no_init<float>> f32_conv_buf;

    int cur_split = -1;
    std::ofstream fout;
    auto close_ofstream = [&]() {
        // Write metadata and close file handler
        if (fout.is_open()) {
            fout.seekp(0);
            std::vector<uint8_t> data(gguf_get_meta_size(ctx_outs[cur_split].get()));
            gguf_get_meta_data(ctx_outs[cur_split].get(), data.data());
            fout.write((const char *) data.data(), data.size());
            fout.close();
        }
    };
    auto new_ofstream = [&](int index) {
        cur_split = index;
        GGML_ASSERT(ctx_outs[cur_split] && "Find uninitialized gguf_context");
        std::string fname = fname_out;
        if (params->keep_split) {
            std::vector<char> split_path(llama_path_max(), 0);
            llama_split_path(split_path.data(), split_path.size(), fname_out.c_str(), cur_split, n_split);
            fname = std::string(split_path.data());
        }

        fout = std::ofstream(fname, std::ios::binary);
        fout.exceptions(std::ofstream::failbit); // fail fast on write errors
        const size_t meta_size = gguf_get_meta_size(ctx_outs[cur_split].get());
        // placeholder for the meta data
        ::zeros(fout, meta_size);
    };

    // no output file for --dry-run
    if (!params->dry_run) {
        new_ofstream(0);
    }

    //
    // main loop: iterate over all weights
    //

    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto & weight = *tensors[i];
        const auto & tm = metadata[i];
        ggml_tensor * tensor = weight.tensor;

        if (!params->dry_run && (weight.idx != cur_split && params->keep_split)) {
            close_ofstream();
            new_ofstream(weight.idx);
        }

        const size_t tensor_size = ggml_nbytes(tensor);

        if (!params->dry_run) {
            if (!ml.use_mmap) {
                if (read_data.size() < tensor_size) {
                    read_data.resize(tensor_size);
                }
                tensor->data = read_data.data();
            }
            ml.load_data_for(tensor);
        }

        LLAMA_LOG_INFO("[%4d/%4d] %-36s - [%s], type = %6s, ",
               ++idx, ml.n_tensors,
               ggml_get_name(tensor),
               llama_format_tensor_shape(tensor).c_str(),
               ggml_type_name(tensor->type));

        const ggml_type cur_type = tensor->type;
        const ggml_type new_type = tm.target_type;

        // If we've decided to quantize to the same type the tensor is already
        // in then there's nothing to do.
        bool quantize = cur_type != new_type;

        void * new_data;
        size_t new_size;

        if (params->dry_run) {
            // the --dry-run option calculates the final quantization size without quantizing
            if (quantize) {
                new_size = ggml_nrows(tensor) * ggml_row_size(new_type, tensor->ne[0]);
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB (%s)\n",
                               tensor_size/1024.0/1024.0,
                               new_size/1024.0/1024.0,
                               ggml_type_name(new_type));
                if (!will_require_imatrix && tm.requires_imatrix) {
                    will_require_imatrix = true;
                }
            } else {
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", new_size/1024.0/1024.0);
            }
            if ((params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
                        params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) && tm.allows_quantization) {
                LLAMA_LOG_INFO("%s: mixed-guided %-36s bucket=%-6s source=%-9s score=%10.6g similarity=%s anchor=%s selected=%7s estimated_size=%8.2f MiB%s",
                        __func__,
                        ggml_get_name(tensor),
                        sensitivity_bucket_name(tm.sensitivity),
                        sensitivity_source_name(tm.sensitivity_from),
                        tm.sensitivity_score,
                        layer_similarity_bucket_name(tm.layer_similarity),
                        anchor_reason_name(tm.anchor_reason).c_str(),
                        ggml_type_name(new_type),
                        new_size/1024.0/1024.0,
                        params->spqr_block_report ? "" : "\n");
                if (params->spqr_block_report) {
                    LLAMA_LOG_INFO(" blocks=%d high=%d medium=%d low=%d\n",
                            tm.block_count, tm.block_high, tm.block_medium, tm.block_low);
                }
            }
            if (params->rd_guided && tm.rd_type != GGML_TYPE_COUNT) {
                LLAMA_LOG_INFO("%s: rd-guided    %-36s selected=%-7s distortion=%10.6g bpw=%6.3f cost=%10.6g\n",
                        __func__, ggml_get_name(tensor), ggml_type_name(new_type),
                        tm.rd_distortion, tm.rd_bpw, tm.rd_cost);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;
            continue;
        } else {
            // no --dry-run, perform quantization
            if (!quantize) {
                new_data = tensor->data;
                new_size = tensor_size;
                LLAMA_LOG_INFO("size = %8.3f MiB\n", tensor_size/1024.0/1024.0);
            } else {
                const int64_t nelements = ggml_nelements(tensor);

                const float * imatrix = nullptr;
                if (imatrix_data) {
                    auto it = imatrix_data->find(tm.remapped_imatrix_name);
                    if (it == imatrix_data->end()) {
                        LLAMA_LOG_INFO("\n====== %s: did not find weights for %s\n", __func__, tensor->name);
                    } else {
                        if (it->second.size() == (size_t)tensor->ne[0]*tensor->ne[2]) {
                            imatrix = it->second.data();
                        } else {
                            LLAMA_LOG_INFO("\n====== %s: imatrix size %d is different from tensor size %d for %s\n", __func__,
                                    int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name);

                            // this can happen when quantizing an old mixtral model with split tensors with a new incompatible imatrix
                            // this is a significant error and it may be good idea to abort the process if this happens,
                            // since many people will miss the error and not realize that most of the model is being quantized without an imatrix
                            // tok_embd should be ignored in this case, since it always causes this warning
                            if (!tensor_name_match_token_embd(tensor->name)) {
                                throw std::runtime_error(format("imatrix size %d is different from tensor size %d for %s",
                                        int(it->second.size()), int(tensor->ne[0]*tensor->ne[2]), tensor->name));
                            }
                        }
                    }
                }
                if (!imatrix && tm.requires_imatrix) {
                    LLAMA_LOG_ERROR("\n\n============================================================\n");
                    LLAMA_LOG_ERROR("Missing importance matrix for tensor %s in a very low-bit quantization\n", tensor->name);
                    LLAMA_LOG_ERROR("The result will be garbage, so bailing out\n");
                    LLAMA_LOG_ERROR("============================================================\n\n");
                    throw std::runtime_error(format("Missing importance matrix for tensor %s in a very low-bit quantization", tensor->name));
                }

                float * f32_data;

                if (tensor->type == GGML_TYPE_F32) {
                    f32_data = (float *) tensor->data;
                } else if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                    throw std::runtime_error(format("requantizing from type %s is disabled", ggml_type_name(tensor->type)));
                } else {
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                LLAMA_LOG_INFO("converting to %s .. ", ggml_type_name(new_type));
                fflush(stdout);

                if (work.size() < (size_t)nelements * 4) {
                    work.resize(nelements * 4); // upper bound on size
                }
                new_data = work.data();

                const int64_t n_per_row = tensor->ne[0];
                const int64_t nrows = tensor->ne[1];

                static const int64_t min_chunk_size = 32 * 512;
                const int64_t chunk_size = (n_per_row >= min_chunk_size ? n_per_row : n_per_row * ((min_chunk_size + n_per_row - 1)/n_per_row));

                const int64_t nelements_matrix = tensor->ne[0] * tensor->ne[1];
                const int64_t nchunk = (nelements_matrix + chunk_size - 1)/chunk_size;
                const int64_t nthread_use = nthread > 1 ? std::max((int64_t)1, std::min((int64_t)nthread, nchunk)) : 1;

                // quantize each expert separately since they have different importance matrices
                new_size = 0;
                for (int64_t i03 = 0; i03 < tensor->ne[2]; ++i03) {
                    const float * f32_data_03 = f32_data + i03 * nelements_matrix;
                    void * new_data_03 = (char *)new_data + ggml_row_size(new_type, n_per_row) * i03 * nrows;
                    const float * imatrix_03 = imatrix ? imatrix + i03 * n_per_row : nullptr;

                    new_size += llama_tensor_quantize_impl(new_type, f32_data_03, new_data_03, chunk_size, nrows, n_per_row, imatrix_03, workers, nthread_use);
                }
                LLAMA_LOG_INFO("size = %8.2f MiB -> %8.2f MiB\n", tensor_size/1024.0/1024.0, new_size/1024.0/1024.0);
            }
            if ((params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
                        params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) && tm.allows_quantization) {
                LLAMA_LOG_INFO("%s: mixed-guided %-36s bucket=%-6s source=%-9s score=%10.6g similarity=%s anchor=%s selected=%7s estimated_size=%8.2f MiB%s",
                        __func__,
                        ggml_get_name(tensor),
                        sensitivity_bucket_name(tm.sensitivity),
                        sensitivity_source_name(tm.sensitivity_from),
                        tm.sensitivity_score,
                        layer_similarity_bucket_name(tm.layer_similarity),
                        anchor_reason_name(tm.anchor_reason).c_str(),
                        ggml_type_name(new_type),
                        new_size/1024.0/1024.0,
                        params->spqr_block_report ? "" : "\n");
                if (params->spqr_block_report) {
                    LLAMA_LOG_INFO(" blocks=%d high=%d medium=%d low=%d\n",
                            tm.block_count, tm.block_high, tm.block_medium, tm.block_low);
                }
            }
            if (params->rd_guided && tm.rd_type != GGML_TYPE_COUNT) {
                LLAMA_LOG_INFO("%s: rd-guided    %-36s selected=%-7s distortion=%10.6g bpw=%6.3f cost=%10.6g\n",
                        __func__, ggml_get_name(tensor), ggml_type_name(new_type),
                        tm.rd_distortion, tm.rd_bpw, tm.rd_cost);
            }
            total_size_org += tensor_size;
            total_size_new += new_size;

            // update the gguf meta data as we go
            gguf_set_tensor_type(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_type);
            GGML_ASSERT(gguf_get_tensor_size(ctx_outs[cur_split].get(), gguf_find_tensor(ctx_outs[cur_split].get(), metadata[i].name.c_str())) == new_size);
            gguf_set_tensor_data(ctx_outs[cur_split].get(), metadata[i].name.c_str(), new_data);

            // write tensor data + padding
            fout.write((const char *) new_data, new_size);
            zeros(fout, GGML_PAD(new_size, align) - new_size);
        } // no --dry-run
    } // main loop

    if (!params->dry_run) {
        close_ofstream();
    }

    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) {
        LLAMA_LOG_INFO("%s: spqr-guided summary: high=%d medium=%d low=%d block_scored=%d promoted=%d total_size=%8.2f MiB avg_bpw=%.2f\n",
                __func__,
                qs.n_spqr_high,
                qs.n_spqr_medium,
                qs.n_spqr_low,
                qs.n_spqr_block_scored,
                qs.n_spqr_promoted,
                total_size_new/1024.0/1024.0,
                total_size_new*8.0/ml.n_elements);
    }
    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) {
        LLAMA_LOG_INFO("%s: layer-delta-guided summary: high_similarity=%d medium_similarity=%d low_similarity=%d anchors=%d anchor_tensors=%d demoted=%d\n",
                __func__, qs.n_delta_high, qs.n_delta_medium, qs.n_delta_low,
                qs.n_anchor_layers, qs.n_anchor_tensors, qs.n_delta_demoted);
    }
    if (params->rd_guided) {
        LLAMA_LOG_INFO("%s: sampled rate-distortion summary: selected=%d lambda=%.6g sample_rows=%d total_size=%8.2f MiB avg_bpw=%.2f\n",
                __func__, qs.n_rd_selected, params->rd_lambda, params->rd_sample_rows,
                total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);
    }

    if (!params->imatrix && params->dry_run && will_require_imatrix) {
        LLAMA_LOG_WARN("%s: WARNING: dry run completed successfully, but actually completing this quantization will require an imatrix!\n",
                       __func__
        );
    }

    if (qs.n_fallback > 0) {
        LLAMA_LOG_WARN("%s: WARNING: %d of %d tensor(s) required fallback quantization\n",
                __func__, qs.n_fallback, ml.n_tensors);
    }
}

//
// interface implementation
//

llama_model_quantize_params llama_model_quantize_default_params() {
    llama_model_quantize_params result = {
        /*.nthread                     =*/ 0,
        /*.ftype                       =*/ LLAMA_FTYPE_MOSTLY_Q8_0,
        /*.output_tensor_type          =*/ GGML_TYPE_COUNT,
        /*.token_embedding_type        =*/ GGML_TYPE_COUNT,
        /*.allow_requantize            =*/ false,
        /*.quantize_output_tensor      =*/ true,
        /*.only_copy                   =*/ false,
        /*.pure                        =*/ false,
        /*.keep_split                  =*/ false,
        /*.dry_run                     =*/ false,
        /*.imatrix                     =*/ nullptr,
        /*.kv_overrides                =*/ nullptr,
        /*.tt_overrides                =*/ nullptr,
        /*.prune_layers                =*/ nullptr,
        /*.mixed_quant_policy          =*/ LLAMA_MIXED_QUANT_POLICY_NONE,
        /*.spqr_block_report           =*/ false,
        /*.spqr_block_size             =*/ 256,
        /*.print_layer_delta_report    =*/ false,
        /*.spqr_block_scoring          =*/ false,
        /*.adaptive_anchors            =*/ false,
        /*.anchor_percentile           =*/ 90.0f,
        /*.print_anchor_report         =*/ false,
        /*.rd_guided                   =*/ false,
        /*.rd_lambda                   =*/ 0.002f,
        /*.rd_sample_rows              =*/ 8,
        /*.print_rd_report             =*/ false,
    };

    return result;
}

uint32_t llama_model_quantize(
        const char * fname_inp,
        const char * fname_out,
        const llama_model_quantize_params * params) {
    try {
        llama_model_quantize_impl(fname_inp, fname_out, params);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to quantize: %s\n", __func__, err.what());
        return 1;
    }

    return 0;
}

//
// Helper functions for external tools exposed in llama-ext.h
//

quantize_state_impl * llama_quant_init(
        const llama_model * model,
        const llama_model_quantize_params * params) {
    return new quantize_state_impl(*model, params);
}

void llama_quant_free(quantize_state_impl * qs) {
    delete qs;
}

llama_model * llama_quant_model_from_metadata(const llama_quant_model_desc * desc) {
    struct llama_model_params mparams = llama_model_default_params();
    auto arch = llm_arch_from_string(desc->architecture);
    auto * model = llama_model_create(arch, mparams);
    model->arch = arch;

    // infer llm_type: only LLM_TYPE_70B matters for quantization logic
    if (model->arch == LLM_ARCH_LLAMA && desc->n_layer == 80 && desc->n_head != desc->n_head_kv) {
        model->type = LLM_TYPE_70B;
    }

    model->hparams.n_embd             = desc->n_embd;
    model->hparams.n_embd_head_k_full = desc->n_embd_head_k;
    model->hparams.n_embd_head_v_full = desc->n_embd_head_v;
    model->hparams.n_layer            = desc->n_layer;
    model->hparams.n_expert           = desc->n_expert;

    for (uint32_t i = 0; i < desc->n_layer; i++) {
        model->hparams.n_head_arr[i]    = desc->n_head;
        model->hparams.n_head_kv_arr[i] = desc->n_head_kv;
        model->hparams.n_ff_arr[i]      = desc->n_ff;
    }

    return model;
}

bool llama_quant_tensor_allows_quantization(
        const quantize_state_impl * qs,
        const ggml_tensor * tensor) {
    return tensor_allows_quantization(qs->params, qs->model.arch, tensor);
}

void llama_quant_compute_types(
        quantize_state_impl * qs,
        llama_ftype ftype,
        ggml_tensor ** tensors,
        ggml_type * result_types,
        size_t n_tensors) {
    // reset per-computation state
    qs->n_attention_wv      = 0;
    qs->n_ffn_down          = 0;
    qs->n_ffn_gate          = 0;
    qs->n_ffn_up            = 0;
    qs->i_attention_wv      = 0;
    qs->i_ffn_down          = 0;
    qs->i_ffn_gate          = 0;
    qs->i_ffn_up            = 0;
    qs->n_fallback          = 0;
    qs->has_imatrix         = false;
    qs->has_tied_embeddings = true;
    qs->n_spqr_low          = 0;
    qs->n_spqr_medium       = 0;
    qs->n_spqr_high         = 0;
    qs->n_spqr_promoted     = 0;
    qs->n_spqr_block_scored = 0;
    qs->n_delta_low         = 0;
    qs->n_delta_medium      = 0;
    qs->n_delta_high        = 0;
    qs->n_delta_demoted     = 0;
    qs->n_anchor_layers     = 0;
    qs->n_anchor_tensors    = 0;
    qs->n_rd_selected       = 0;

    // build metadata from tensor names
    std::vector<tensor_metadata> metadata(n_tensors);
    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].name = ggml_get_name(tensors[i]);
    }

    // initialize counters and categories
    init_quantize_state_counters(*qs, metadata);

    // use a local copy of params with the requested ftype
    llama_model_quantize_params local_params = *qs->params;
    local_params.ftype = ftype;

    for (size_t i = 0; i < n_tensors; i++) {
        metadata[i].allows_quantization = tensor_allows_quantization(&local_params, qs->model.arch, tensors[i]);
        metadata[i].remapped_imatrix_name = metadata[i].name;
    }
    if (local_params.mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            local_params.mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) {
        init_spqr_guided_sensitivity(*qs, metadata, nullptr);
        if (local_params.spqr_block_report || local_params.spqr_block_scoring) {
            init_spqr_guided_block_scoring(*qs, metadata, nullptr, local_params.spqr_block_size, local_params.spqr_block_scoring);
        }
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
