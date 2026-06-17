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

static bool spqr_layer_delta_guidance_enabled(const llama_model_quantize_params * params) {
    return params->layer_delta_guidance ||
           params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA;
}

static float teacher_feature_gate_mix(const llama_model_quantize_params * params) {
    return params->quant_teacher_aware ? 0.10f : 0.0f;
}

struct logit_gate_metrics {
    bool paired = false;
    bool has_damage = false;
    bool has_kl = false;
    bool has_flip = false;
    float damage_score = 0.0f;
    float mean_topk_kl = 0.0f;
    float argmax_flip_rate = 0.0f;
    std::unordered_map<std::string, float> tensor_delta_mean_mse;
    std::unordered_map<std::string, float> tensor_delta_confidence;
    std::unordered_map<int, float> layer_delta_mean_mse;
    std::unordered_map<int, float> layer_delta_confidence;
    std::unordered_map<std::string, float> family_delta_mean_mse;
    std::unordered_map<std::string, float> family_delta_confidence;
};

static float logit_attribution_confidence(int compared_tensors, int compared_values) {
    const float tensor_cov = std::min(1.0f, compared_tensors / 4.0f);
    const float value_cov = std::min(1.0f, std::sqrt(std::max(0, compared_values) / 4096.0f));
    return std::clamp(0.25f + 0.35f * tensor_cov + 0.40f * value_cov, 0.25f, 1.0f);
}

static bool parse_logit_metric(const std::string & json, const char * key, float & value) {
    const std::regex pattern(std::string("\"") + key + "\"\\s*:\\s*([-+0-9.eE]+)");
    std::smatch match;
    if (!std::regex_search(json, match, pattern) || match.size() < 2) {
        return false;
    }
    try {
        value = std::stof(match[1].str());
    } catch (...) {
        return false;
    }
    return std::isfinite(value);
}

static bool extract_named_json_array(const std::string & json, const char * key, std::string & out) {
    const std::string needle = std::string("\"") + key + "\"";
    const size_t key_pos = json.find(needle);
    if (key_pos == std::string::npos) {
        return false;
    }

    const size_t array_start = json.find('[', key_pos + needle.size());
    if (array_start == std::string::npos) {
        return false;
    }

    int depth = 0;
    for (size_t i = array_start; i < json.size(); ++i) {
        const char c = json[i];
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                out = json.substr(array_start, i - array_start + 1);
                return true;
            }
        }
    }

    return false;
}

static void parse_logit_attribution_deltas(
        const std::string & json,
        const char * key,
        std::unordered_map<std::string, float> * tensor_out,
        std::unordered_map<std::string, float> * tensor_confidence_out,
        std::unordered_map<int, float> * layer_out,
        std::unordered_map<int, float> * layer_confidence_out,
        std::unordered_map<std::string, float> * family_out,
        std::unordered_map<std::string, float> * family_confidence_out) {
    std::string array_json;
    if (!extract_named_json_array(json, key, array_json)) {
        return;
    }

    const std::regex entry_pattern(
        R"json(\{\s*"key"\s*:\s*"([^"]+)"\s*,\s*"layer"\s*:\s*(-?\d+)\s*,(?:\s*"compared_tensors"\s*:\s*(\d+)\s*,\s*"compared_values"\s*:\s*(\d+)\s*,)?[\s\S]*?"delta_mean_mse"\s*:\s*([-+0-9.eE]+))json");

    for (std::sregex_iterator it(array_json.begin(), array_json.end(), entry_pattern), end;
            it != end; ++it) {
        const std::smatch & match = *it;
        if (match.size() < 6) {
            continue;
        }

        try {
            const std::string name = match[1].str();
            const int layer = std::stoi(match[2].str());
            const int compared_tensors = match[3].matched ? std::stoi(match[3].str()) : 4;
            const int compared_values = match[4].matched ? std::stoi(match[4].str()) : 4096;
            const float delta = std::stof(match[5].str());
            if (!std::isfinite(delta)) {
                continue;
            }
            const float confidence = logit_attribution_confidence(compared_tensors, compared_values);
            if (layer_out != nullptr && layer >= 0) {
                (*layer_out)[layer] = delta;
            }
            if (layer_confidence_out != nullptr && layer >= 0) {
                (*layer_confidence_out)[layer] = confidence;
            }
            if (tensor_out != nullptr) {
                (*tensor_out)[name] = delta;
            }
            if (tensor_confidence_out != nullptr) {
                (*tensor_confidence_out)[name] = confidence;
            }
            if (family_out != nullptr) {
                (*family_out)[name] = delta;
            }
            if (family_confidence_out != nullptr) {
                (*family_confidence_out)[name] = confidence;
            }
        } catch (...) {
            continue;
        }
    }
}

static bool load_logit_gate_metrics(const char * path, logit_gate_metrics & metrics) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (json.empty()) {
        return false;
    }

    metrics.has_damage = parse_logit_metric(json, "damage_score", metrics.damage_score);
    metrics.has_kl = parse_logit_metric(json, "delta_mean_topk_kl", metrics.mean_topk_kl);
    metrics.has_flip = parse_logit_metric(json, "delta_argmax_flip_rate", metrics.argmax_flip_rate);
    metrics.paired = metrics.has_damage || metrics.has_kl || metrics.has_flip;

    if (!metrics.has_kl) {
        metrics.has_kl = parse_logit_metric(json, "mean_topk_kl", metrics.mean_topk_kl);
    }
    if (!metrics.has_flip) {
        metrics.has_flip = parse_logit_metric(json, "argmax_flip_rate", metrics.argmax_flip_rate);
    }

    if (metrics.paired) {
        parse_logit_attribution_deltas(json, "tensor_attribution_delta",
                &metrics.tensor_delta_mean_mse, &metrics.tensor_delta_confidence,
                nullptr, nullptr, nullptr, nullptr);
        parse_logit_attribution_deltas(json, "layer_attribution_delta",
                nullptr, nullptr,
                &metrics.layer_delta_mean_mse, &metrics.layer_delta_confidence,
                nullptr, nullptr);
        parse_logit_attribution_deltas(json, "family_attribution_delta",
                nullptr, nullptr,
                nullptr, nullptr,
                &metrics.family_delta_mean_mse, &metrics.family_delta_confidence);
    }

    return metrics.has_damage || metrics.has_kl || metrics.has_flip;
}

static void ensure_logit_gate_loaded(quantize_state_impl & qs);

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
    float rd_allocation_lambda = 0.0f;
    float rd_budget_distortion_base = 0.0f;
    float rd_budget_distortion_selected = 0.0f;
    size_t rd_target_bytes = 0;
    size_t rd_estimated_bytes = 0;
    bool rd_quality_limited = false;
    bool logit_report_loaded = false;
    bool logit_report_available = false;
    bool logit_gate_pass = true;
    bool logit_report_paired = false;
    float logit_damage_score = 0.0f;
    float logit_mean_topk_kl = 0.0f;
    float logit_argmax_flip_rate = 0.0f;
    std::unordered_map<std::string, float> logit_tensor_delta_mean_mse;
    std::unordered_map<std::string, float> logit_tensor_delta_confidence;
    std::unordered_map<int, float> logit_layer_delta_mean_mse;
    std::unordered_map<int, float> logit_layer_delta_confidence;
    std::unordered_map<std::string, float> logit_family_delta_mean_mse;
    std::unordered_map<std::string, float> logit_family_delta_confidence;

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

static void ensure_logit_gate_loaded(quantize_state_impl & qs) {
    if (qs.logit_report_loaded) {
        return;
    }
    qs.logit_report_loaded = true;

    const llama_model_quantize_params * params = qs.params;
    if (params->logit_report == nullptr || params->logit_report[0] == '\0') {
        return;
    }

    logit_gate_metrics metrics;
    if (!load_logit_gate_metrics(params->logit_report, metrics)) {
        LLAMA_LOG_WARN("%s: logit gate could not parse report %s; continuing without a global logit gate\n",
                __func__, params->logit_report);
        return;
    }

    qs.logit_report_available = true;
    qs.logit_report_paired = metrics.paired;
    qs.logit_damage_score = metrics.damage_score;
    qs.logit_mean_topk_kl = metrics.mean_topk_kl;
    qs.logit_argmax_flip_rate = metrics.argmax_flip_rate;
    qs.logit_tensor_delta_mean_mse = std::move(metrics.tensor_delta_mean_mse);
    qs.logit_tensor_delta_confidence = std::move(metrics.tensor_delta_confidence);
    qs.logit_layer_delta_mean_mse = std::move(metrics.layer_delta_mean_mse);
    qs.logit_layer_delta_confidence = std::move(metrics.layer_delta_confidence);
    qs.logit_family_delta_mean_mse = std::move(metrics.family_delta_mean_mse);
    qs.logit_family_delta_confidence = std::move(metrics.family_delta_confidence);
    qs.logit_gate_pass = true;
    if (params->logit_gate) {
        qs.logit_gate_pass =
            (!metrics.has_damage || metrics.damage_score <= params->logit_damage_threshold) &&
            (!metrics.has_kl || metrics.mean_topk_kl <= params->logit_kl_threshold) &&
            (!metrics.has_flip || metrics.argmax_flip_rate <= params->logit_flip_threshold);
    }

    LLAMA_LOG_INFO("%s: logit gate report=%s mode=%s damage=%10.6g kl=%10.6g flips=%10.6g thresholds=(%g,%g,%g) status=%s tensor_deltas=%zu layer_deltas=%zu family_deltas=%zu\n",
            __func__, params->logit_report,
            qs.logit_report_paired ? "paired" : "candidate-only",
            qs.logit_damage_score,
            qs.logit_mean_topk_kl,
            qs.logit_argmax_flip_rate,
            params->logit_damage_threshold,
            params->logit_kl_threshold,
            params->logit_flip_threshold,
            params->logit_gate ? (qs.logit_gate_pass ? "pass" : "fail") : "report-only",
            qs.logit_tensor_delta_mean_mse.size(),
            qs.logit_layer_delta_mean_mse.size(),
            qs.logit_family_delta_mean_mse.size());
}

static void quant_log_section(const char * title) {
    LLAMA_LOG_INFO("\n==== %s ====\n", title);
}

// A normal GGUF tensor-type candidate. Future compression backends can participate in
// the same global allocator by providing an equivalent rate-distortion candidate.
struct rd_candidate {
    ggml_type type = GGML_TYPE_COUNT;
    float weighted_distortion = 0.0f;
    float distortion = 0.0f;
    float bpw = 0.0f;
    size_t bytes = 0;
};

struct spqr_repair_eval {
    ggml_type type = GGML_TYPE_COUNT;
    float weighted_mse = 0.0f;
    float gain_error = 0.0f;
    float shape_error = 0.0f;
    float outlier_concentration = 0.0f;
    float composite_error = 0.0f;
    float bpw = 0.0f;
};

struct teacher_gate_eval {
    float proxy_error = -1.0f;
    float block_error = 0.0f;
    float rank_error = 0.0f;
    float feature_l1_error = 0.0f;
    float feature_cosine_error = 0.0f;
    float feature_norm_error = 0.0f;
    float feature_error = 0.0f;
    float gate_error = 0.0f;
};

struct teacher_gate_extra {
    float block_error = 0.0f;
    float rank_error = 0.0f;
    float feature_l1_error = 0.0f;
    float feature_cosine_error = 0.0f;
    float feature_norm_error = 0.0f;
    float feature_error = 0.0f;
};

struct compression_opportunity {
    size_t tensor_index = 0;
    std::string name;
    ggml_type selected_type = GGML_TYPE_COUNT;
    ggml_type candidate_type = GGML_TYPE_COUNT;
    size_t saved_bytes = 0;
    float selected_error = 0.0f;
    float selected_gate_error = 0.0f;
    float candidate_error = 0.0f;
    float candidate_gate_error = 0.0f;
    float candidate_block_error = 0.0f;
    float candidate_rank_error = 0.0f;
    float candidate_feature_error = 0.0f;
    float teacher_risk = 1.0f;
    float teacher_risk_tensor = 0.0f;
    float teacher_risk_layer = 0.0f;
    float teacher_risk_family = 0.0f;
    float teacher_safe_pressure = 0.0f;
    float teacher_promotion_pressure = 0.0f;
    float budget_bias = 1.0f;
    float repaired_candidate_error = 0.0f;
    float effective_candidate_error = 0.0f;
    float candidate_weighted_mse = 0.0f;
    float delta_error = 0.0f;
    float opportunity_score = 0.0f;
    float selected_bpw = 0.0f;
    float candidate_bpw = 0.0f;
    bool proxy_safe = false;
    bool repair_potential = false;
};

struct promotion_opportunity {
    size_t tensor_index = 0;
    std::string name;
    ggml_type selected_type = GGML_TYPE_COUNT;
    ggml_type candidate_type = GGML_TYPE_COUNT;
    size_t extra_bytes = 0;
    float selected_error = 0.0f;
    float selected_gate_error = 0.0f;
    float candidate_error = 0.0f;
    float candidate_gate_error = 0.0f;
    float candidate_block_error = 0.0f;
    float candidate_rank_error = 0.0f;
    float candidate_feature_error = 0.0f;
    float candidate_weighted_mse = 0.0f;
    float teacher_risk = 1.0f;
    float teacher_risk_tensor = 0.0f;
    float teacher_risk_layer = 0.0f;
    float teacher_risk_family = 0.0f;
    float teacher_safe_pressure = 0.0f;
    float teacher_promotion_pressure = 0.0f;
    float quality_bias = 1.0f;
    float damage_reduction = 0.0f;
    float opportunity_score = 0.0f;
    float selected_bpw = 0.0f;
    float candidate_bpw = 0.0f;
};

struct promotion_package {
    promotion_opportunity promotion;
    std::vector<compression_opportunity> demotions;
    size_t saved_bytes = 0;
    float package_damage = 0.0f;
    float overshoot_mib = 0.0f;
    float correlation_penalty = 0.0f;
    float uncertainty_penalty = 0.0f;
    float net_gain = 0.0f;
    float package_score = 0.0f;
    bool valid = false;
};

static bool compression_opportunity_better(const compression_opportunity & a, const compression_opportunity & b) {
    if (a.proxy_safe != b.proxy_safe) {
        return a.proxy_safe > b.proxy_safe;
    }
    if (a.repair_potential != b.repair_potential) {
        return a.repair_potential > b.repair_potential;
    }
    if (a.opportunity_score != b.opportunity_score) {
        return a.opportunity_score > b.opportunity_score;
    }
    return a.saved_bytes > b.saved_bytes;
}

static bool promotion_opportunity_better(const promotion_opportunity & a, const promotion_opportunity & b) {
    if (a.opportunity_score != b.opportunity_score) {
        return a.opportunity_score > b.opportunity_score;
    }
    if (a.damage_reduction != b.damage_reduction) {
        return a.damage_reduction > b.damage_reduction;
    }
    return a.extra_bytes < b.extra_bytes;
}

static bool promotion_package_better(const promotion_package & a, const promotion_package & b) {
    if (a.valid != b.valid) {
        return a.valid > b.valid;
    }
    if (a.package_score != b.package_score) {
        return a.package_score > b.package_score;
    }
    if (a.net_gain != b.net_gain) {
        return a.net_gain > b.net_gain;
    }
    if (a.promotion.opportunity_score != b.promotion.opportunity_score) {
        return a.promotion.opportunity_score > b.promotion.opportunity_score;
    }
    return a.promotion.extra_bytes < b.promotion.extra_bytes;
}

struct spqr_teacher_repair_result {
    float clip_abs = 0.0f;
    float source_gain = 1.0f;
    float base_error = 0.0f;
    float repaired_error = 0.0f;
    float improvement = 0.0f;
    bool gain_repaired = false;
};

// per-tensor metadata, computed in the preliminary loop and used in the main loop
struct tensor_metadata {
    std::string     name;
    ggml_type       target_type;
    ggml_type       requested_type;
    ggml_type       compatibility_fallback_type;
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
    std::vector<rd_candidate> rd_candidates;
    float           rd_refinement_score;
    bool            rd_from_profile;
    float           activity_mean;
    float           activity_variance;
    float           activity_peak_ratio;
    float           activity_active_fraction;
    float           activity_risk;
    float           activity_shift;
    std::string     remapped_imatrix_name;
    bool            allows_quantization;
    bool            requires_imatrix;
    bool            had_shape_fallback;
    int64_t         fallback_ncols;
    int64_t         fallback_required_block_size;
    float           teacher_repair_clip_abs;
    float           teacher_repair_source_gain;
    float           teacher_repair_error_before;
    float           teacher_repair_error_after;
};

struct quant_budget_state {
    size_t target_bytes = 0;
    size_t current_bytes = 0;
};

struct repair_gate_settings {
    float accept_ratio = 0.0f;
    float max_error = 0.0f;
    float min_error = 0.0f;
    float min_improvement = 0.0f;
};

static size_t estimate_quantized_tensor_bytes(
        const ggml_tensor * tensor,
        ggml_type target_type) {
    return target_type == tensor->type ?
        ggml_nbytes(tensor) :
        (size_t) ggml_nrows(tensor) * ggml_row_size(target_type, tensor->ne[0]);
}

static size_t estimate_quantized_model_bytes(
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        const std::vector<tensor_metadata> & metadata) {
    size_t total = 0;
    for (size_t i = 0; i < metadata.size(); ++i) {
        total += estimate_quantized_tensor_bytes(tensors[i]->tensor, metadata[i].target_type);
    }
    return total;
}

static size_t quant_target_bytes(
        const quantize_state_impl & qs,
        const llama_model_quantize_params * params,
        const llama_model_loader & ml) {
    if (qs.rd_target_bytes > 0) {
        return qs.rd_target_bytes;
    }
    return params->rd_target_bpw > 0.0f ?
        (size_t) std::ceil(params->rd_target_bpw * ml.n_elements / 8.0) :
        (size_t) std::ceil(params->rd_target_size_mib * 1024.0 * 1024.0);
}

static quant_budget_state quant_budget_snapshot(
        const quantize_state_impl & qs,
        const llama_model_quantize_params * params,
        const llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        const std::vector<tensor_metadata> & metadata) {
    return {
        quant_target_bytes(qs, params, ml),
        estimate_quantized_model_bytes(tensors, metadata),
    };
}

static void reset_teacher_repair_state(tensor_metadata & tm) {
    tm.teacher_repair_clip_abs = 0.0f;
    tm.teacher_repair_source_gain = 1.0f;
    tm.teacher_repair_error_before = 0.0f;
    tm.teacher_repair_error_after = 0.0f;
}

static void apply_quant_transition(
        tensor_metadata & tm,
        ggml_type type,
        float weighted_mse,
        float distortion_weight,
        float bpw,
        float rd_cost) {
    tm.target_type = type;
    tm.rd_type = type;
    tm.rd_distortion = weighted_mse / std::max(1e-20f, distortion_weight);
    tm.rd_bpw = bpw;
    tm.rd_cost = rd_cost;
    reset_teacher_repair_state(tm);
}

static repair_gate_settings repair_gate_for_context(
        const quantize_state_impl & qs,
        bool budget_limited,
        bool include_minimums) {
    const llama_model_quantize_params * params = qs.params;
    repair_gate_settings settings;
    settings.accept_ratio = budget_limited ?
        std::max(params->quant_repair_accept_ratio, 1.25f) :
        params->quant_repair_accept_ratio;
    settings.max_error = budget_limited ?
        std::max(params->quant_repair_max_error, 0.005f) :
        params->quant_repair_max_error;
    settings.min_error = budget_limited ?
        params->quant_repair_min_error * 0.05f :
        params->quant_repair_min_error;
    settings.min_improvement = budget_limited ?
        std::min(params->quant_repair_min_improvement, 0.001f) :
        params->quant_repair_min_improvement;

    if (params->logit_gate && qs.logit_report_available && !qs.logit_gate_pass) {
        settings.accept_ratio = std::min(settings.accept_ratio, budget_limited ? 1.10f : 1.01f);
        settings.max_error = std::min(settings.max_error, budget_limited ? 0.0025f : 0.0005f);
        if (include_minimums) {
            settings.min_error *= budget_limited ? 1.5f : 1.25f;
            settings.min_improvement = std::max(settings.min_improvement, budget_limited ? 0.01f : 0.08f);
        }
    }
    return settings;
}

static float budget_layer_position_bias(const tensor_metadata & tm, int max_layer) {
    if (tm.layer < 0 || max_layer <= 0) {
        return 1.0f;
    }
    // Mild bottom-first compression prior, inspired by local feature-distillation
    // work such as Lillama (arXiv:2412.16719): when a hard budget forces a tie,
    // prefer saving bytes in earlier transformer layers and protect later layers.
    const float pos = (float) tm.layer / std::max(1, max_layer);
    if (pos <= 0.25f) {
        return 1.08f;
    }
    if (pos >= 0.75f) {
        return 0.94f;
    }
    return 1.0f;
}

static float quality_layer_position_bias(const tensor_metadata & tm, int max_layer) {
    if (tm.layer < 0 || max_layer <= 0) {
        return 1.0f;
    }
    const float pos = (float) tm.layer / std::max(1, max_layer);
    if (pos <= 0.25f) {
        return 0.94f;
    }
    if (pos >= 0.75f) {
        return 1.08f;
    }
    return 1.0f;
}

static bool demotion_package_conflicts_with_promotion(
        const compression_opportunity & demotion,
        const promotion_opportunity & promotion,
        const std::vector<tensor_metadata> & metadata) {
    if (demotion.tensor_index == promotion.tensor_index) {
        return true;
    }
    const tensor_metadata & demotion_tm = metadata[demotion.tensor_index];
    const tensor_metadata & promotion_tm = metadata[promotion.tensor_index];
    if (demotion_tm.layer >= 0 && promotion_tm.layer >= 0 && demotion_tm.layer == promotion_tm.layer) {
        return true;
    }
    return false;
}

static float evaluate_promotion_package(
        const llama_model_quantize_params * params,
        const promotion_opportunity & promotion,
        const std::vector<compression_opportunity> & demotions,
        const std::vector<tensor_metadata> & metadata,
        size_t required_financing_bytes,
        promotion_package & out) {
    size_t saved_bytes = 0;
    float package_damage = 0.0f;
    float uncertainty_penalty = 0.0f;
    std::unordered_map<int, int> layer_counts;
    std::unordered_map<int, int> adjacent_counts;
    std::unordered_map<int, int> category_counts;

    for (const compression_opportunity & demotion : demotions) {
        saved_bytes += demotion.saved_bytes;
        package_damage += demotion.delta_error;
        if (!demotion.proxy_safe) {
            uncertainty_penalty += 0.25f * demotion.delta_error;
        }
        const tensor_metadata & tm = metadata[demotion.tensor_index];
        if (tm.layer >= 0) {
            layer_counts[tm.layer] += 1;
            adjacent_counts[tm.layer / 2] += 1;
        }
        category_counts[(int) tm.category] += 1;
    }

    if (saved_bytes < required_financing_bytes) {
        return -std::numeric_limits<float>::infinity();
    }

    float correlation_penalty = 0.0f;
    for (const auto & [layer, count] : layer_counts) {
        GGML_UNUSED(layer);
        if (count >= 2) {
            correlation_penalty += params->logit_search_package_same_layer_penalty * (float) (count - 1);
        }
    }
    for (const auto & [group, count] : adjacent_counts) {
        GGML_UNUSED(group);
        if (count >= 2) {
            correlation_penalty += 0.5f * params->logit_search_package_same_layer_penalty * (float) (count - 1);
        }
    }
    for (const auto & [category, count] : category_counts) {
        GGML_UNUSED(category);
        if (count >= 2) {
            correlation_penalty += 0.4f * params->logit_search_package_same_layer_penalty * (float) (count - 1);
        }
    }

    const float overshoot_mib =
        saved_bytes > required_financing_bytes ?
        (float) (saved_bytes - required_financing_bytes) / 1024.0f / 1024.0f :
        0.0f;
    const float overshoot_penalty = params->logit_search_package_overshoot_weight * overshoot_mib;
    const float total_package_cost = package_damage + uncertainty_penalty + correlation_penalty + overshoot_penalty;
    const float net_gain = promotion.damage_reduction - total_package_cost;
    const float package_score = net_gain / std::max(
            (float) promotion.extra_bytes / 1024.0f / 1024.0f,
            1e-9f);

    out.promotion = promotion;
    out.demotions = demotions;
    out.saved_bytes = saved_bytes;
    out.package_damage = package_damage;
    out.overshoot_mib = overshoot_mib;
    out.correlation_penalty = correlation_penalty;
    out.uncertainty_penalty = uncertainty_penalty;
    out.net_gain = net_gain;
    out.package_score = package_score;
    out.valid = net_gain > std::max(0.00005f, 0.10f * promotion.damage_reduction);
    return package_score;
}

static int max_transformer_layer(const std::vector<tensor_metadata> & metadata) {
    int max_layer = -1;
    for (const tensor_metadata & tm : metadata) {
        max_layer = std::max(max_layer, tm.layer);
    }
    return max_layer;
}

static const char * tensor_logit_family_key(tensor_category category) {
    switch (category) {
        case tensor_category::ATTENTION_Q:
        case tensor_category::ATTENTION_K:
        case tensor_category::ATTENTION_V:
        case tensor_category::ATTENTION_QKV:
        case tensor_category::ATTENTION_KV_B:
        case tensor_category::ATTENTION_OUTPUT:
            return "attn_out";
        case tensor_category::FFN_UP:
        case tensor_category::FFN_GATE:
        case tensor_category::FFN_DOWN:
            return "ffn_out";
        default:
            return nullptr;
    }
}

static std::string tensor_logit_group_key(const tensor_metadata & tm) {
    if (tm.layer < 0) {
        return {};
    }
    if (const char * family = tensor_logit_family_key(tm.category)) {
        return std::string(family) + "-" + std::to_string(tm.layer);
    }
    return {};
}

static float lookup_logit_tensor_confidence(const quantize_state_impl & qs, const tensor_metadata & tm) {
    auto it = qs.logit_tensor_delta_confidence.find(tm.name);
    if (it != qs.logit_tensor_delta_confidence.end()) {
        return it->second;
    }
    const std::string tensor_group = tensor_logit_group_key(tm);
    if (!tensor_group.empty()) {
        it = qs.logit_tensor_delta_confidence.find(tensor_group);
        if (it != qs.logit_tensor_delta_confidence.end()) {
            return it->second;
        }
    }
    return 1.0f;
}

static float lookup_logit_layer_confidence(const quantize_state_impl & qs, int layer) {
    auto it = qs.logit_layer_delta_confidence.find(layer);
    return it != qs.logit_layer_delta_confidence.end() ? it->second : 1.0f;
}

static float lookup_logit_family_confidence(const quantize_state_impl & qs, const std::string & family) {
    auto it = qs.logit_family_delta_confidence.find(family);
    return it != qs.logit_family_delta_confidence.end() ? it->second : 1.0f;
}

static float logit_delta_to_risk_component(float delta, float scale) {
    if (!std::isfinite(delta) || delta == 0.0f) {
        return 0.0f;
    }
    const float signed_log = std::copysign(std::log1pf(std::fabs(delta)), delta);
    return scale * signed_log;
}

struct logit_teacher_risk_breakdown {
    float total = 1.0f;
    float tensor = 0.0f;
    float layer = 0.0f;
    float family = 0.0f;
};

static logit_teacher_risk_breakdown logit_local_teacher_risk_breakdown(
        const quantize_state_impl & qs,
        const tensor_metadata & tm) {
    logit_teacher_risk_breakdown out;
    if (!qs.logit_report_available || !qs.logit_report_paired) {
        return out;
    }

    const std::string tensor_group = tensor_logit_group_key(tm);
    auto tensor_it = qs.logit_tensor_delta_mean_mse.find(tm.name);
    if (tensor_it == qs.logit_tensor_delta_mean_mse.end() && !tensor_group.empty()) {
        tensor_it = qs.logit_tensor_delta_mean_mse.find(tensor_group);
    }

    const bool has_tensor_delta = tensor_it != qs.logit_tensor_delta_mean_mse.end();
    const float tensor_confidence = lookup_logit_tensor_confidence(qs, tm);
    const float tensor_scale = has_tensor_delta ? 0.28f : 0.18f;
    const float layer_scale = has_tensor_delta ? 0.06f : 0.12f;
    const float l_out_scale = has_tensor_delta ? 0.025f : 0.05f;
    const float family_scale = has_tensor_delta ? 0.04f : 0.08f;

    if (tensor_it != qs.logit_tensor_delta_mean_mse.end()) {
        out.tensor += tensor_confidence * logit_delta_to_risk_component(tensor_it->second, tensor_scale);
    }

    if (tm.layer >= 0) {
        auto it = qs.logit_layer_delta_mean_mse.find(tm.layer);
        if (it != qs.logit_layer_delta_mean_mse.end()) {
            const float layer_confidence = lookup_logit_layer_confidence(qs, tm.layer);
            out.layer += layer_confidence * logit_delta_to_risk_component(it->second, layer_scale);
        }

        auto l_out = qs.logit_family_delta_mean_mse.find("l_out");
        if (l_out != qs.logit_family_delta_mean_mse.end()) {
            const float l_out_confidence = lookup_logit_family_confidence(qs, "l_out");
            out.family += l_out_confidence * logit_delta_to_risk_component(l_out->second, l_out_scale);
        }
    }

    if (const char * family = tensor_logit_family_key(tm.category)) {
        auto it = qs.logit_family_delta_mean_mse.find(family);
        if (it != qs.logit_family_delta_mean_mse.end()) {
            const float family_confidence = lookup_logit_family_confidence(qs, family);
            out.family += family_confidence * logit_delta_to_risk_component(it->second, family_scale);
        }
    }

    const float risk_cap = has_tensor_delta ? 1.60f : 1.45f;
    out.total = std::clamp(1.0f + out.tensor + out.layer + out.family, 0.80f, risk_cap);
    return out;
}

static float logit_local_teacher_risk(const quantize_state_impl & qs, const tensor_metadata & tm) {
    return logit_local_teacher_risk_breakdown(qs, tm).total;
}

static float logit_positive_risk(float x) {
    return std::max(0.0f, x);
}

static float logit_negative_risk(float x) {
    return std::max(0.0f, -x);
}

static float logit_local_promotion_pressure(const logit_teacher_risk_breakdown & risk) {
    const float weighted =
        1.75f * logit_positive_risk(risk.tensor) +
        0.75f * logit_positive_risk(risk.layer) +
        0.50f * logit_positive_risk(risk.family);
    return std::clamp(weighted, 0.0f, 1.0f);
}

static float logit_local_safe_compression_pressure(const logit_teacher_risk_breakdown & risk) {
    const float weighted =
        1.25f * logit_negative_risk(risk.tensor) +
        0.50f * logit_negative_risk(risk.layer) +
        0.25f * logit_negative_risk(risk.family);
    return std::clamp(weighted, 0.0f, 1.0f);
}

static float logit_local_demotion_risk(const logit_teacher_risk_breakdown & risk) {
    const float promotion_pressure = logit_local_promotion_pressure(risk);
    const float safe_pressure = logit_local_safe_compression_pressure(risk);
    return std::clamp(risk.total + 0.35f * promotion_pressure - 0.20f * safe_pressure, 0.70f, 1.90f);
}

static float logit_local_accept_ratio(float base_accept_ratio, float risk) {
    return 1.0f + (base_accept_ratio - 1.0f) / std::max(0.80f, risk);
}

static float logit_local_error_limit(float base_max_error, float risk) {
    return base_max_error / std::max(0.80f, risk);
}

static bool tensor_type_can_convert_to_f32(ggml_type type) {
    if (type == GGML_TYPE_F32 ||
            type == GGML_TYPE_F16 ||
            type == GGML_TYPE_BF16) {
        return true;
    }
    if (!ggml_is_quantized(type)) {
        return false;
    }
    const ggml_type_traits * traits = ggml_get_type_traits(type);
    return traits && traits->to_float;
}

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
    if (!tensor_type_can_convert_to_f32(tensor->type) || tensor->type == GGML_TYPE_F32) {
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
        ml.load_data_for(tensor);
    } else if (tensor->data == nullptr) {
        ml.load_data_for(tensor);
    }

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

static float robust_scene_threshold(const std::vector<float> & scores, float percentile_value) {
    if (scores.empty()) {
        return 0.0f;
    }
    const float percentile_threshold = percentile(scores, percentile_value);
    const float median = percentile(scores, 0.5f);
    std::vector<float> deviations;
    deviations.reserve(scores.size());
    for (float score : scores) {
        deviations.push_back(std::abs(score - median));
    }
    const float mad = percentile(deviations, 0.5f);
    return std::max(percentile_threshold, median + 2.5f * 1.4826f * mad);
}

// MPEG-inspired activity masking and scene-change signals. These only steer existing
// per-tensor quant types; they do not introduce dynamic bit allocation at inference.
static std::map<int, float> layer_scene_scores(
        const std::map<int, float> & avg_rel_by_layer,
        const std::vector<tensor_metadata> & metadata) {
    std::map<int, std::vector<float>> shifts_by_layer;
    for (const tensor_metadata & tm : metadata) {
        if (tm.layer >= 0 && tensor_category_participates_in_layer_delta(tm.category)) {
            shifts_by_layer[tm.layer].push_back(tm.activity_shift);
        }
    }

    std::map<int, float> result;
    for (const auto & kv : avg_rel_by_layer) {
        float avg_shift = 0.0f;
        auto shift_it = shifts_by_layer.find(kv.first);
        if (shift_it != shifts_by_layer.end() && !shift_it->second.empty()) {
            avg_shift = std::accumulate(shift_it->second.begin(), shift_it->second.end(), 0.0f) / shift_it->second.size();
        }
        result[kv.first] = kv.second * (1.0f + 0.5f * avg_shift);
    }
    return result;
}

static void init_activity_profile(std::vector<tensor_metadata> & metadata, const llama_model_quantize_params * params) {
    if (!params->activity_profile) {
        return;
    }

    std::unordered_map<std::string, const llama_model_quantize_activity_data *> profiles;
    for (const llama_model_quantize_activity_data * p = params->activity_profile; p->name != nullptr; ++p) {
        profiles.emplace(p->name, p);
    }

    std::map<std::pair<int, int>, tensor_metadata *> by_layer_category;
    int loaded = 0;
    for (tensor_metadata & tm : metadata) {
        tm.layer = tensor_get_layer_index(tm.name);
        auto it = profiles.find(tm.name);
        if (it == profiles.end()) {
            continue;
        }
        tm.activity_mean = it->second->mean;
        tm.activity_variance = it->second->variance;
        tm.activity_peak_ratio = it->second->peak_ratio;
        tm.activity_active_fraction = it->second->active_fraction;

        const float cv = std::sqrt(std::max(0.0f, tm.activity_variance)) / (tm.activity_mean + 1e-20f);
        const float peak_signal = std::log1p(std::max(0.0f, tm.activity_peak_ratio));
        const float rare_active = (1.0f - std::clamp(tm.activity_active_fraction, 0.0f, 1.0f)) * std::min(peak_signal, 3.0f) / 3.0f;
        tm.activity_risk = std::clamp(1.0f + 0.12f * peak_signal + 0.08f * std::log1p(cv) + 0.15f * rare_active, 1.0f, 1.75f);
        if (tm.layer >= 0) {
            by_layer_category[{ tm.layer, (int) tm.category }] = &tm;
        }
        ++loaded;
    }

    for (auto & kv : by_layer_category) {
        tensor_metadata * cur = kv.second;
        auto prev_it = by_layer_category.find({ cur->layer - 1, (int) cur->category });
        if (prev_it == by_layer_category.end()) {
            continue;
        }
        const tensor_metadata * prev = prev_it->second;
        const float mean_change = std::abs(std::log((cur->activity_mean + 1e-20f) / (prev->activity_mean + 1e-20f)));
        const float peak_change = std::abs(std::log1p(cur->activity_peak_ratio) - std::log1p(prev->activity_peak_ratio));
        const float active_change = std::abs(cur->activity_active_fraction - prev->activity_active_fraction);
        cur->activity_shift = std::clamp(0.50f * mean_change + 0.25f * peak_change + 0.25f * active_change, 0.0f, 2.0f);
    }

    LLAMA_LOG_INFO("%s: loaded activity-mask statistics for %d tensor(s)\n", __func__, loaded);
}

static int init_precomputed_layer_delta_analysis(
        quantize_state_impl & qs,
        std::vector<tensor_metadata> & metadata,
        bool print_report) {
    if (!qs.params->layer_delta_profile) {
        return 0;
    }

    std::unordered_map<std::string, const llama_model_quantize_layer_delta_data *> profiles;
    for (const llama_model_quantize_layer_delta_data * p = qs.params->layer_delta_profile; p->name != nullptr; ++p) {
        profiles.emplace(p->name, p);
    }

    int expected = 0;
    int matched = 0;
    for (tensor_metadata & tm : metadata) {
        tm.layer = tensor_get_layer_index(tm.name);
        if (tm.allows_quantization && tm.layer > 0 && tensor_category_participates_in_layer_delta(tm.category)) {
            ++expected;
            matched += profiles.find(tm.name) != profiles.end();
        }
    }
    if (matched < expected) {
        LLAMA_LOG_INFO("%s: layer-delta profile coverage is partial (%d/%d); falling back to full analysis\n",
                __func__, matched, expected);
        return 0;
    }

    int loaded = 0;
    std::map<int, std::vector<float>> rel_by_layer;
    for (tensor_metadata & tm : metadata) {
        auto it = profiles.find(tm.name);
        if (it == profiles.end() || !tensor_category_participates_in_layer_delta(tm.category)) {
            continue;
        }

        tm.rel_delta_norm = it->second->rel_delta_norm;
        tm.cosine_similarity = it->second->cosine_similarity;
        tm.layer_similarity = layer_similarity_bucket_from_metrics(tm.rel_delta_norm, tm.cosine_similarity);
        switch (tm.layer_similarity) {
            case layer_similarity_bucket::LOW:    ++qs.n_delta_low;    break;
            case layer_similarity_bucket::MEDIUM: ++qs.n_delta_medium; break;
            case layer_similarity_bucket::HIGH:   ++qs.n_delta_high;   break;
            case layer_similarity_bucket::UNKNOWN: break;
        }
        rel_by_layer[tm.layer].push_back(tm.rel_delta_norm);
        ++loaded;

        if (print_report) {
            LLAMA_LOG_INFO("%s: layer-delta-profile layer=%4d tensor=%-36s rel_delta_norm=%10.6f cosine_sim=%10.6f sensitivity=%-6s similarity=%s\n",
                    __func__, tm.layer, tm.name.c_str(), tm.rel_delta_norm, tm.cosine_similarity,
                    sensitivity_bucket_name(tm.sensitivity), layer_similarity_bucket_name(tm.layer_similarity));
        }
    }

    if (loaded > 0 && qs.params->adaptive_anchors) {
        std::map<int, float> avg_rel_by_layer;
        for (const auto & kv : rel_by_layer) {
            const float avg = std::accumulate(kv.second.begin(), kv.second.end(), 0.0f) / kv.second.size();
            avg_rel_by_layer[kv.first] = avg;
        }
        const std::map<int, float> scene_scores = layer_scene_scores(avg_rel_by_layer, metadata);
        std::vector<float> layer_scores;
        for (const auto & kv : scene_scores) {
            layer_scores.push_back(kv.second);
        }

        std::map<int, int> anchor_reasons;
        const int n_layer = (int) qs.model.hparams.n_layer;
        if (n_layer > 0) {
            anchor_reasons[0] |= ANCHOR_REASON_FIRST_LAYER;
            anchor_reasons[n_layer - 1] |= ANCHOR_REASON_FINAL_LAYER;
        }
        float scene_threshold = 0.0f;
        if (!layer_scores.empty()) {
            scene_threshold = robust_scene_threshold(layer_scores, qs.params->anchor_percentile / 100.0f);
            for (const auto & kv : scene_scores) {
                if (kv.second >= scene_threshold) {
                    anchor_reasons[kv.first] |= ANCHOR_REASON_SCENE_CHANGE;
                }
            }
        }
        for (tensor_metadata & tm : metadata) {
            auto it = anchor_reasons.find(tm.layer);
            if (it != anchor_reasons.end() && tensor_category_participates_in_layer_delta(tm.category)) {
                tm.anchor_reason = it->second;
                ++qs.n_anchor_tensors;
            }
        }
        qs.n_anchor_layers = (int) anchor_reasons.size();
        LLAMA_LOG_INFO("%s: adaptive anchors from profile: layers=%d tensors=%d percentile=%.1f robust_scene_threshold=%.6f\n",
                __func__, qs.n_anchor_layers, qs.n_anchor_tensors, qs.params->anchor_percentile, scene_threshold);
        if (qs.params->print_anchor_report) {
            for (const auto & kv : anchor_reasons) {
                const auto rel_it = avg_rel_by_layer.find(kv.first);
                const auto score_it = scene_scores.find(kv.first);
                LLAMA_LOG_INFO("%s: anchor-profile layer=%4d avg_rel_delta=%10.6f scene_score=%10.6f reason=%s\n",
                        __func__, kv.first, rel_it != avg_rel_by_layer.end() ? rel_it->second : 0.0f,
                        score_it != scene_scores.end() ? score_it->second : 0.0f,
                        anchor_reason_name(kv.second).c_str());
            }
        }
    }

    LLAMA_LOG_INFO("%s: reused %d precomputed layer-delta entries from imatrix analysis profile\n", __func__, loaded);
    return loaded;
}

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
        if (tm.allows_quantization && tm.layer >= 0 &&
                ggml_n_dims(tensors[i]->tensor) >= 2 &&
                tensor_category_participates_in_layer_delta(tm.category)) {
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
        if (!tm.allows_quantization || tm.layer <= 0 ||
                ggml_n_dims(tensors[i]->tensor) < 2 ||
                !tensor_category_participates_in_layer_delta(tm.category)) {
            continue;
        }

        auto prev_it = layer_category_to_index.find({ tm.layer - 1, (int) tm.category });
        if (prev_it == layer_category_to_index.end()) {
            continue;
        }

        ggml_tensor * cur_tensor = tensors[i]->tensor;
        ggml_tensor * prev_tensor = tensors[prev_it->second]->tensor;
        if (!ggml_are_same_shape(cur_tensor, prev_tensor)) {
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

        const std::map<int, float> scene_scores = layer_scene_scores(avg_rel_by_layer, metadata);
        std::vector<float> layer_scores;
        layer_scores.reserve(scene_scores.size());
        for (const auto & kv : scene_scores) {
            layer_scores.push_back(kv.second);
        }

        float scene_threshold = 0.0f;
        if (!layer_scores.empty()) {
            scene_threshold = robust_scene_threshold(layer_scores, qs.params->anchor_percentile / 100.0f);
            for (const auto & kv : scene_scores) {
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

        LLAMA_LOG_INFO("%s: adaptive anchors: layers=%d tensors=%d percentile=%.1f robust_scene_threshold=%.6f\n",
                __func__, qs.n_anchor_layers, qs.n_anchor_tensors, qs.params->anchor_percentile, scene_threshold);
        if (qs.params->print_anchor_report) {
            for (const auto & kv : anchor_reasons) {
                const auto rel_it = avg_rel_by_layer.find(kv.first);
                const auto score_it = scene_scores.find(kv.first);
                LLAMA_LOG_INFO("%s: anchor layer=%4d avg_rel_delta=%10.6f scene_score=%10.6f reason=%s\n",
                        __func__,
                        kv.first,
                        rel_it != avg_rel_by_layer.end() ? rel_it->second : 0.0f,
                        score_it != scene_scores.end() ? score_it->second : 0.0f,
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

static std::string format_shape_fallback_reason(int64_t ncols, int64_t required_block_size) {
    return format("incompatible_ncols_%" PRId64 "_not_multiple_of_%" PRId64, ncols, required_block_size);
}

static std::string format_type_list(const std::vector<ggml_type> & types) {
    std::string result;
    for (size_t i = 0; i < types.size(); ++i) {
        if (i > 0) {
            result += ",";
        }
        result += ggml_type_name(types[i]);
    }
    return result;
}

// Primary K-quant ladder used by the experimental RD-guided mixed-quant path.
// This keeps the coarse candidate space explicit and stable: start from a
// compact Q3/Q4/Q5/Q6 progression, then also evaluate the requested base type
// if it falls outside the ladder.
static std::vector<ggml_type> rd_primary_k_ladder(const llama_model_quantize_params * params, ggml_type base_type) {
    std::vector<ggml_type> result;
    auto add = [&result] (ggml_type type) {
        if (std::find(result.begin(), result.end(), type) == result.end()) {
            result.push_back(type);
        }
    };

    if (params->rd_include_iq3) {
        add(GGML_TYPE_IQ3_S);
    }
    add(GGML_TYPE_Q3_K);
    add(GGML_TYPE_Q4_K);
    add(GGML_TYPE_Q5_K);
    add(GGML_TYPE_Q6_K);
    add(base_type);
    return result;
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
static ggml_type llama_tensor_get_type(quantize_state_impl & qs, const llama_model_quantize_params * params, const ggml_tensor * tensor, ggml_type default_type, tensor_metadata & tm) {
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
            } else if (spqr_layer_delta_guidance_enabled(params)) {
                new_type = spqr_layer_delta_type_for_bucket(
                        params, tm.category, default_type, tm.sensitivity, tm.layer_similarity, tm.anchor_reason != ANCHOR_REASON_NONE);
            } else if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED) {
                new_type = spqr_guided_type_for_bucket(params, tm.category, default_type, tm.sensitivity);
            } else {
                new_type = llama_tensor_get_type_impl(qs, new_type, tensor, params->ftype, tm.category);
            }
        }

        tm.requested_type = new_type;
        tm.had_shape_fallback = false;
        tm.compatibility_fallback_type = new_type;
        tm.fallback_ncols = tensor->ne[0];
        tm.fallback_required_block_size = ggml_blck_size(new_type);

        // incompatible tensor shapes are handled here - fallback to a compatible type
        new_type = tensor_type_fallback(qs, tensor, new_type);
        tm.compatibility_fallback_type = new_type;
        tm.had_shape_fallback = new_type != tm.requested_type;

        if ((params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
                    spqr_layer_delta_guidance_enabled(params)) && !manual) {
            const int64_t ncols = tensor->ne[0];
            const bool can_compare =
                ncols % ggml_blck_size(default_type) == 0 &&
                ncols % ggml_blck_size(new_type) == 0;
            if (can_compare && ggml_row_size(new_type, ncols) > ggml_row_size(default_type, ncols)) {
                ++qs.n_spqr_promoted;
            } else if (spqr_layer_delta_guidance_enabled(params) &&
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
        tm.rd_candidates.clear();
        tm.activity_mean = 0.0f;
        tm.activity_variance = 0.0f;
        tm.activity_peak_ratio = 0.0f;
        tm.activity_active_fraction = 0.0f;
        tm.activity_risk = 1.0f;
        tm.activity_shift = 0.0f;

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
struct rd_block_stats {
    float aggregate = -1.0f;
    float mean = -1.0f;
    float p90 = -1.0f;
    float worst = -1.0f;
};

static rd_block_stats aggregate_rd_block_distortions(std::vector<float> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const float mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    const size_t p90_index = std::min(values.size() - 1, (size_t) std::ceil(0.90 * values.size()) - 1);
    return {
        0.50f * mean + 0.30f * values[p90_index] + 0.20f * values.back(),
        mean,
        values[p90_index],
        values.back(),
    };
}

static rd_block_stats sample_rd_candidate(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type candidate,
        int sample_rows) {
    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    if (!ggml_is_quantized(candidate) ||
            ncols % ggml_blck_size(candidate) != 0 ||
            (ggml_quantize_requires_imatrix(candidate) && !imatrix)) {
        return {};
    }

    const ggml_type_traits * traits = ggml_get_type_traits(candidate);
    if (!traits || !traits->to_float) {
        return {};
    }

    std::vector<no_init<uint8_t>> quantized(ggml_row_size(candidate, ncols));
    std::vector<float> reconstructed(ncols);
    std::vector<float> block_distortions;
    block_distortions.reserve(sample_rows);

    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * src = data.data() + row * ncols;
        const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
        const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;

        ggml_quantize_chunk(candidate, src, quantized.data(), 0, 1, ncols, row_imatrix);
        traits->to_float(quantized.data(), reconstructed.data(), ncols);

        double error_sum = 0.0;
        double signal_sum = 0.0;
        for (int64_t col = 0; col < ncols; ++col) {
            const double weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0;
            const double delta = (double) src[col] - reconstructed[col];
            error_sum += weight * delta * delta;
            signal_sum += weight * src[col] * src[col];
        }
        block_distortions.push_back((float) (error_sum / (signal_sum + 1e-20)));
    }

    return aggregate_rd_block_distortions(std::move(block_distortions));
}

static float sample_teacher_proxy_error(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type candidate,
        int sample_rows,
        float clip_abs,
        float source_gain,
        const std::vector<float> * channel_weights = nullptr,
        float output_proxy_mix = 0.0f) {
    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    if (!ggml_is_quantized(candidate) ||
            ncols % ggml_blck_size(candidate) != 0 ||
            (ggml_quantize_requires_imatrix(candidate) && !imatrix)) {
        return -1.0f;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(candidate);
    if (!traits || !traits->to_float) {
        return -1.0f;
    }

    std::vector<no_init<uint8_t>> quantized(ggml_row_size(candidate, ncols));
    std::vector<float> source(ncols);
    std::vector<float> reconstructed(ncols);

    double total_error = 0.0;
    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * teacher = data.data() + row * ncols;
        const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
        const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;

        if (clip_abs > 0.0f) {
            for (int64_t col = 0; col < ncols; ++col) {
                source[col] = source_gain * std::max(-clip_abs, std::min(clip_abs, teacher[col]));
            }
        } else {
            for (int64_t col = 0; col < ncols; ++col) {
                source[col] = source_gain * teacher[col];
            }
        }

        ggml_quantize_chunk(candidate, source.data(), quantized.data(), 0, 1, ncols, row_imatrix);
        traits->to_float(quantized.data(), reconstructed.data(), ncols);

        double error_sum = 0.0;
        double signal_sum = 0.0;
        double output_error_sum = 0.0;
        double output_signal_sum = 0.0;
        for (int64_t col = 0; col < ncols; ++col) {
            const double weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0;
            const double channel_weight = channel_weights && (size_t) col < channel_weights->size() ?
                std::max(0.0f, (*channel_weights)[col]) : 1.0;
            const double delta = (double) teacher[col] - reconstructed[col];
            error_sum += weight * delta * delta;
            signal_sum += weight * teacher[col] * teacher[col];
            output_error_sum += weight * channel_weight * delta * delta;
            output_signal_sum += weight * channel_weight * teacher[col] * teacher[col];
        }
        const double base_error = error_sum / (signal_sum + 1e-20);
        if (channel_weights && output_proxy_mix > 0.0f) {
            const double output_error = output_error_sum / (output_signal_sum + 1e-20);
            total_error += (1.0 - output_proxy_mix) * base_error + output_proxy_mix * output_error;
        } else {
            total_error += base_error;
        }
    }

    return (float) (total_error / sample_rows);
}

static teacher_gate_extra sample_teacher_gate_extra(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type candidate,
        int sample_rows,
        int top_k) {
    teacher_gate_extra result;
    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    if (!ggml_is_quantized(candidate) ||
            ncols % ggml_blck_size(candidate) != 0 ||
            (ggml_quantize_requires_imatrix(candidate) && !imatrix)) {
        return result;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(candidate);
    if (!traits || !traits->to_float) {
        return result;
    }

    std::vector<no_init<uint8_t>> quantized(ggml_row_size(candidate, ncols));
    std::vector<float> reconstructed(ncols);
    const int rank_k = std::max(1, std::min(top_k, (int) ncols));

    double dot = 0.0;
    double teacher_norm = 0.0;
    double rec_norm = 0.0;
    double l1_error = 0.0;
    double abs_signal = 0.0;
    double rank_error = 0.0;
    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * teacher = data.data() + row * ncols;
        const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
        const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;

        ggml_quantize_chunk(candidate, teacher, quantized.data(), 0, 1, ncols, row_imatrix);
        traits->to_float(quantized.data(), reconstructed.data(), ncols);

        int64_t teacher_top = 0;
        int64_t rec_top = 0;
        double teacher_top_abs = -1.0;
        double teacher_second_abs = -1.0;
        double rec_top_abs = -1.0;
        double rec_second_abs = -1.0;
        std::vector<std::pair<double, int64_t>> teacher_rank;
        std::vector<std::pair<double, int64_t>> rec_rank;
        teacher_rank.reserve((size_t) ncols);
        rec_rank.reserve((size_t) ncols);

        for (int64_t col = 0; col < ncols; ++col) {
            const double weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0;
            const double teacher_v = teacher[col];
            const double rec_v = reconstructed[col];
            const double delta = teacher_v - rec_v;
            dot += weight * teacher_v * rec_v;
            teacher_norm += weight * teacher_v * teacher_v;
            rec_norm += weight * rec_v * rec_v;
            l1_error += weight * std::abs(delta);
            abs_signal += weight * std::abs(teacher_v);

            const double teacher_abs = weight * std::abs(teacher_v);
            const double rec_abs = weight * std::abs(rec_v);
            teacher_rank.emplace_back(teacher_abs, col);
            rec_rank.emplace_back(rec_abs, col);
            if (teacher_abs > teacher_top_abs) {
                teacher_second_abs = teacher_top_abs;
                teacher_top_abs = teacher_abs;
                teacher_top = col;
            } else if (teacher_abs > teacher_second_abs) {
                teacher_second_abs = teacher_abs;
            }
            if (rec_abs > rec_top_abs) {
                rec_second_abs = rec_top_abs;
                rec_top_abs = rec_abs;
                rec_top = col;
            } else if (rec_abs > rec_second_abs) {
                rec_second_abs = rec_abs;
            }
        }

        const auto keep_top = [] (auto & values, int k) {
            if ((int) values.size() > k) {
                std::nth_element(values.begin(), values.begin() + k, values.end(), std::greater<>());
                values.resize(k);
            }
            std::sort(values.begin(), values.end(), [] (const auto & a, const auto & b) {
                return a.second < b.second;
            });
        };
        keep_top(teacher_rank, rank_k);
        keep_top(rec_rank, rank_k);

        int overlap = 0;
        size_t ti = 0;
        size_t ri = 0;
        while (ti < teacher_rank.size() && ri < rec_rank.size()) {
            if (teacher_rank[ti].second == rec_rank[ri].second) {
                ++overlap;
                ++ti;
                ++ri;
            } else if (teacher_rank[ti].second < rec_rank[ri].second) {
                ++ti;
            } else {
                ++ri;
            }
        }

        const double overlap_loss = 1.0 - (double) overlap / rank_k;
        const double teacher_margin = std::max(0.0, teacher_top_abs - std::max(0.0, teacher_second_abs));
        const double rec_margin = teacher_top == rec_top ?
            std::max(0.0, rec_top_abs - std::max(0.0, rec_second_abs)) :
            0.0;
        const double margin_loss = teacher_margin > 0.0 ?
            std::max(0.0, teacher_margin - rec_margin) / (teacher_top_abs + 1e-20) :
            0.0;
        const double top1_flip = teacher_top == rec_top ? 0.0 : 1.0;
        rank_error += 0.50 * overlap_loss + 0.30 * margin_loss + 0.20 * top1_flip;
    }

    const double cosine = dot / (std::sqrt(teacher_norm * rec_norm) + 1e-20);
    const double cosine_error = std::max(0.0, 1.0 - cosine);
    const double norm_error = std::abs(std::sqrt(teacher_norm) - std::sqrt(rec_norm)) /
        (std::sqrt(teacher_norm) + 1e-20);
    const double normalized_l1_error = l1_error / (abs_signal + 1e-20);
    result.block_error = (float) (0.70 * cosine_error + 0.30 * norm_error);
    result.rank_error = (float) (rank_error / std::max(1, sample_rows));
    result.feature_l1_error = (float) normalized_l1_error;
    result.feature_cosine_error = (float) cosine_error;
    result.feature_norm_error = (float) norm_error;
    result.feature_error = (float) (0.40 * normalized_l1_error + 0.35 * cosine_error + 0.25 * norm_error);
    return result;
}

static bool tensor_name_is_ffn_repair_candidate(const char * name) {
    const std::string tensor_name = name ? name : "";
    return tensor_name.find("ffn_down") != std::string::npos ||
           tensor_name.find("ffn_gate") != std::string::npos ||
           tensor_name.find("ffn_up") != std::string::npos;
}

static std::vector<float> build_ffn_repair_channel_weights(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        int sample_rows) {
    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    std::vector<float> weights;
    if (!tensor_name_is_ffn_repair_candidate(tensor->name) || ncols <= 0 || nrows <= 0) {
        return weights;
    }

    weights.assign(ncols, 0.0f);
    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * src = data.data() + row * ncols;
        const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
        const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;
        for (int64_t col = 0; col < ncols; ++col) {
            const float base_weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0f;
            weights[col] += base_weight * std::abs(src[col]);
        }
    }

    const float inv_samples = sample_rows > 0 ? 1.0f / sample_rows : 1.0f;
    double mean_weight = 0.0;
    for (float & weight : weights) {
        weight *= inv_samples;
        mean_weight += weight;
    }
    mean_weight /= std::max<int64_t>(1, ncols);
    if (mean_weight <= 0.0) {
        std::fill(weights.begin(), weights.end(), 1.0f);
        return weights;
    }

    for (float & weight : weights) {
        const float normalized = weight / (float) mean_weight;
        weight = std::min(4.0f, std::max(0.5f, normalized));
    }
    return weights;
}

static spqr_teacher_repair_result evaluate_teacher_repair_clipping(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type target_type,
        int sample_rows,
        float min_error,
        float min_improvement,
        bool use_clipping,
        bool use_scale) {
    spqr_teacher_repair_result result;
    const std::vector<float> channel_weights = build_ffn_repair_channel_weights(tensor, data, imatrix, sample_rows);
    const float output_proxy_mix = channel_weights.empty() ? 0.0f : 0.35f;
    result.base_error = sample_teacher_proxy_error(
            tensor, data, imatrix, target_type, sample_rows, 0.0f, 1.0f,
            channel_weights.empty() ? nullptr : &channel_weights,
            output_proxy_mix);
    result.repaired_error = result.base_error;
    if (result.base_error < min_error) {
        return result;
    }

    // This approximates layer-output repair with a diagonal activation covariance:
    // E[||X(W-Wq)||^2] is estimated with imatrix weights, then we try exportable
    // source transforms that still write ordinary GGUF tensor types.
    if (use_scale) {
        const float gain_candidates[] = { 0.970f, 0.985f, 0.995f, 1.005f, 1.015f, 1.030f };
        for (float gain : gain_candidates) {
            const float repaired_error = sample_teacher_proxy_error(
                    tensor, data, imatrix, target_type, sample_rows, 0.0f, gain,
                    channel_weights.empty() ? nullptr : &channel_weights,
                    output_proxy_mix);
            if (repaired_error >= 0.0f && repaired_error < result.repaired_error) {
                result.repaired_error = repaired_error;
                result.source_gain = gain;
                result.clip_abs = 0.0f;
                result.gain_repaired = true;
            }
        }
    }

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    if (!use_clipping) {
        if ((result.clip_abs > 0.0f || result.gain_repaired) && result.base_error > 0.0f) {
            result.improvement = (result.base_error - result.repaired_error) / result.base_error;
            if (result.improvement < min_improvement) {
                result.clip_abs = 0.0f;
                result.source_gain = 1.0f;
                result.gain_repaired = false;
                result.repaired_error = result.base_error;
                result.improvement = 0.0f;
            }
        }
        return result;
    }

    std::vector<float> abs_values;
    abs_values.reserve((size_t) sample_rows * (size_t) ncols);
    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * src = data.data() + row * ncols;
        for (int64_t col = 0; col < ncols; ++col) {
            abs_values.push_back(std::abs(src[col]));
        }
    }
    if (abs_values.empty()) {
        return result;
    }

    std::sort(abs_values.begin(), abs_values.end());
    const float percentiles[] = { 0.999f, 0.995f, 0.990f, 0.980f };
    for (float p : percentiles) {
        const size_t index = std::min(abs_values.size() - 1, (size_t) std::floor(p * (abs_values.size() - 1)));
        const float clip_abs = abs_values[index];
        if (clip_abs <= 0.0f) {
            continue;
        }
        const float repaired_error = sample_teacher_proxy_error(
                tensor, data, imatrix, target_type, sample_rows, clip_abs, 1.0f,
                channel_weights.empty() ? nullptr : &channel_weights,
                output_proxy_mix);
        if (repaired_error >= 0.0f && repaired_error < result.repaired_error) {
            result.repaired_error = repaired_error;
            result.clip_abs = clip_abs;
            result.source_gain = 1.0f;
            result.gain_repaired = false;
        }
        if (use_scale) {
            const float gain_candidates[] = { 0.985f, 0.995f, 1.005f, 1.015f };
            for (float gain : gain_candidates) {
                const float gain_error = sample_teacher_proxy_error(
                        tensor, data, imatrix, target_type, sample_rows, clip_abs, gain,
                        channel_weights.empty() ? nullptr : &channel_weights,
                        output_proxy_mix);
                if (gain_error >= 0.0f && gain_error < result.repaired_error) {
                    result.repaired_error = gain_error;
                    result.clip_abs = clip_abs;
                    result.source_gain = gain;
                    result.gain_repaired = gain != 1.0f;
                }
            }
        }
    }

    if ((result.clip_abs > 0.0f || result.gain_repaired) && result.base_error > 0.0f) {
        result.improvement = (result.base_error - result.repaired_error) / result.base_error;
        if (result.improvement < min_improvement) {
            result.clip_abs = 0.0f;
            result.source_gain = 1.0f;
            result.gain_repaired = false;
        }
    }
    return result;
}

static teacher_gate_eval evaluate_teacher_gate_error(
        const llama_model_quantize_params * params,
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type candidate,
        int sample_rows,
        float local_error) {
    teacher_gate_eval result;
    result.gate_error = local_error;
    if (!params->quant_teacher_aware || params->quant_teacher_aware_mix <= 0.0f) {
        return result;
    }

    const float proxy_error = sample_teacher_proxy_error(tensor, data, imatrix, candidate, sample_rows, 0.0f, 1.0f);
    if (proxy_error < 0.0f) {
        return result;
    }

    const float mix = std::min(1.0f, std::max(0.0f, params->quant_teacher_aware_mix));
    result.proxy_error = proxy_error;
    result.gate_error = (1.0f - mix) * local_error + mix * proxy_error;
    if (params->quant_teacher_aware_block_mix > 0.0f ||
            params->quant_teacher_aware_rank_mix > 0.0f ||
            teacher_feature_gate_mix(params) > 0.0f) {
        const teacher_gate_extra extra = sample_teacher_gate_extra(
                tensor, data, imatrix, candidate, sample_rows,
                std::max(1, params->quant_teacher_aware_top_k));
        result.block_error = extra.block_error;
        result.rank_error = extra.rank_error;
        result.feature_l1_error = extra.feature_l1_error;
        result.feature_cosine_error = extra.feature_cosine_error;
        result.feature_norm_error = extra.feature_norm_error;
        result.feature_error = extra.feature_error;
        result.gate_error += params->quant_teacher_aware_block_mix * result.block_error;
        result.gate_error += params->quant_teacher_aware_rank_mix * result.rank_error;
        result.gate_error += teacher_feature_gate_mix(params) * result.feature_error;
    }
    return result;
}

static float rd_distortion_weight(const tensor_metadata & tm) {
    float weight = 1.0f;
    if (tm.sensitivity == sensitivity_bucket::HIGH) {
        weight = 2.0f;
    } else if (tm.sensitivity == sensitivity_bucket::MEDIUM) {
        weight = 1.35f;
    }
    if (tm.layer_similarity == layer_similarity_bucket::LOW) {
        weight *= 1.20f;
    } else if (tm.layer_similarity == layer_similarity_bucket::HIGH) {
        weight *= 0.85f;
    }
    return weight * tm.activity_risk;
}

static bool is_scalar_fallback_type(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 ||
           type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0;
}

static std::vector<ggml_type> spqr_repair_candidate_types(
        const llama_model_quantize_params * params,
        const ggml_tensor * tensor,
        ggml_type selected_type) {
    const int64_t ncols = tensor->ne[0];
    std::vector<ggml_type> candidates;
    auto add = [&] (ggml_type type) {
        if (type != selected_type &&
                ncols % ggml_blck_size(type) == 0 &&
                ggml_row_size(type, ncols) < ggml_row_size(selected_type, ncols) &&
                std::find(candidates.begin(), candidates.end(), type) == candidates.end()) {
            candidates.push_back(type);
        }
    };

    add(GGML_TYPE_Q6_K);
    add(GGML_TYPE_Q5_K);
    if (params->rd_include_iq3) {
        add(GGML_TYPE_IQ3_S);
    }
    add(GGML_TYPE_Q5_1);
    add(GGML_TYPE_Q5_0);
    add(GGML_TYPE_Q4_K);
    add(GGML_TYPE_Q4_1);
    add(GGML_TYPE_Q4_0);
    add(GGML_TYPE_Q3_K);
    return candidates;
}

static std::vector<ggml_type> spqr_repair_promotion_candidate_types(
        const llama_model_quantize_params * params,
        const ggml_tensor * tensor,
        ggml_type selected_type) {
    GGML_UNUSED(params);
    const int64_t ncols = tensor->ne[0];
    std::vector<ggml_type> candidates;
    auto add = [&] (ggml_type type) {
        if (type != selected_type &&
                ncols % ggml_blck_size(type) == 0 &&
                ggml_row_size(type, ncols) > ggml_row_size(selected_type, ncols) &&
                std::find(candidates.begin(), candidates.end(), type) == candidates.end()) {
            candidates.push_back(type);
        }
    };

    add(GGML_TYPE_Q4_K);
    add(GGML_TYPE_Q5_0);
    add(GGML_TYPE_Q5_1);
    add(GGML_TYPE_Q5_K);
    add(GGML_TYPE_Q6_K);
    add(GGML_TYPE_Q8_0);
    return candidates;
}

static bool quant_budget_limited(const llama_model_quantize_params * params) {
    return params->rd_target_bpw > 0.0f || params->rd_target_size_mib > 0.0f;
}

static spqr_repair_eval evaluate_spqr_repair_candidate(
        const ggml_tensor * tensor,
        const std::vector<float> & data,
        const float * imatrix,
        ggml_type candidate,
        int sample_rows,
        float distortion_weight) {
    spqr_repair_eval result;
    result.type = candidate;

    const int64_t ncols = tensor->ne[0];
    const int64_t nrows = ggml_nrows(tensor);
    if (!ggml_is_quantized(candidate) ||
            ncols % ggml_blck_size(candidate) != 0 ||
            (ggml_quantize_requires_imatrix(candidate) && !imatrix)) {
        result.type = GGML_TYPE_COUNT;
        return result;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(candidate);
    if (!traits || !traits->to_float) {
        result.type = GGML_TYPE_COUNT;
        return result;
    }

    std::vector<no_init<uint8_t>> quantized(ggml_row_size(candidate, ncols));
    std::vector<float> reconstructed(ncols);
    std::vector<double> weighted_errors;
    weighted_errors.reserve(ncols);

    double weighted_mse = 0.0;
    double gain_error = 0.0;
    double shape_error = 0.0;
    double outlier_concentration = 0.0;

    for (int sample = 0; sample < sample_rows; ++sample) {
        const int64_t row = sample_rows == 1 ? 0 : sample * (nrows - 1) / (sample_rows - 1);
        const float * src = data.data() + row * ncols;
        const int64_t expert = tensor->ne[1] > 0 ? row / tensor->ne[1] : 0;
        const float * row_imatrix = imatrix ? imatrix + expert * ncols : nullptr;

        ggml_quantize_chunk(candidate, src, quantized.data(), 0, 1, ncols, row_imatrix);
        traits->to_float(quantized.data(), reconstructed.data(), ncols);

        double error_sum = 0.0;
        double signal_sum = 0.0;
        double src_norm = 0.0;
        double rec_norm = 0.0;
        double dot = 0.0;
        weighted_errors.clear();
        for (int64_t col = 0; col < ncols; ++col) {
            const double weight = row_imatrix ? std::max(0.0f, row_imatrix[col]) : 1.0;
            const double src_v = src[col];
            const double rec_v = reconstructed[col];
            const double delta = src_v - rec_v;
            const double weighted_error = weight * delta * delta;
            weighted_errors.push_back(weighted_error);
            error_sum += weighted_error;
            signal_sum += weight * src_v * src_v;
            src_norm += src_v * src_v;
            rec_norm += rec_v * rec_v;
            dot += src_v * rec_v;
        }

        weighted_mse += error_sum / (signal_sum + 1e-20);
        gain_error += std::abs(std::sqrt(src_norm) - std::sqrt(rec_norm)) / (std::sqrt(src_norm) + 1e-20);
        const double cosine = dot / (std::sqrt(src_norm * rec_norm) + 1e-20);
        shape_error += std::max(0.0, 1.0 - cosine);

        std::sort(weighted_errors.begin(), weighted_errors.end(), std::greater<double>());
        const size_t top_n = std::max<size_t>(1, weighted_errors.size() / 20);
        const double top_error = std::accumulate(weighted_errors.begin(), weighted_errors.begin() + top_n, 0.0);
        outlier_concentration += error_sum > 0.0 ? top_error / error_sum : 0.0;
    }

    result.weighted_mse = distortion_weight * (float) (weighted_mse / sample_rows);
    result.gain_error = (float) (gain_error / sample_rows);
    result.shape_error = (float) (shape_error / sample_rows);
    result.outlier_concentration = (float) (outlier_concentration / sample_rows);
    result.composite_error = result.weighted_mse + 0.05f * result.gain_error + 0.05f * result.shape_error;
    result.bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
    return result;
}

static std::vector<ggml_type> scalar_fallback_candidate_types(const tensor_metadata & tm, ggml_type original_type) {
    if (tm.category == tensor_category::TOKEN_EMBD || tm.category == tensor_category::OUTPUT) {
        return { GGML_TYPE_Q8_0 };
    }

    switch (original_type) {
        case GGML_TYPE_Q4_0:
            return { GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 };
        case GGML_TYPE_Q5_0:
            return { GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 };
        case GGML_TYPE_Q5_1:
            return { GGML_TYPE_Q5_1, GGML_TYPE_Q8_0 };
        case GGML_TYPE_Q8_0:
            return { GGML_TYPE_Q8_0 };
        default:
            return { original_type };
    }
}

static void apply_shape_aware_scalar_fallback(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->rd_guided || params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int evaluated = 0;
    int changed = 0;
    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);

        if (!tm.allows_quantization || !is_scalar_fallback_type(tm.target_type) ||
                ncols <= 0 || nrows <= 0 || ncols % 256 == 0) {
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

        const ggml_type original_type = tm.target_type;
        const ggml_type requested_type = tm.requested_type == GGML_TYPE_COUNT ? original_type : tm.requested_type;
        const float distortion_weight = rd_distortion_weight(tm);
        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        ggml_type best_type = GGML_TYPE_COUNT;
        float best_cost = std::numeric_limits<float>::infinity();
        float best_distortion = 0.0f;
        float best_bpw = 0.0f;

        for (ggml_type candidate : scalar_fallback_candidate_types(tm, original_type)) {
            const rd_block_stats stats = sample_rd_candidate(tensor, data, imatrix, candidate, sample_rows);
            if (stats.aggregate < 0.0f) {
                continue;
            }
            const float bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
            const float cost = distortion_weight * stats.aggregate + params->rd_lambda * bpw;
            if (cost < best_cost) {
                best_cost = cost;
                best_type = candidate;
                best_distortion = stats.aggregate;
                best_bpw = bpw;
            }
            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: scalar-fallback-candidate tensor=%-36s shape=%" PRId64 "x%" PRId64 " type=%-7s distortion=%10.6g weight=%5.2f bpw=%6.3f cost=%10.6g\n",
                        __func__, tm.name.c_str(), nrows, ncols, ggml_type_name(candidate),
                        stats.aggregate, distortion_weight, bpw, cost);
            }
        }

        if (best_type == GGML_TYPE_COUNT) {
            continue;
        }

        ++evaluated;
        tm.target_type = best_type;
        if (tm.rd_type == original_type || tm.rd_type == GGML_TYPE_COUNT) {
            tm.rd_type = best_type;
            tm.rd_distortion = best_distortion;
            tm.rd_bpw = best_bpw;
            tm.rd_cost = best_cost;
        }
        changed += best_type != original_type;

        const std::vector<ggml_type> effective_candidates = scalar_fallback_candidate_types(tm, original_type);
        const std::string reason = tm.had_shape_fallback ?
                format_shape_fallback_reason(tm.fallback_ncols, tm.fallback_required_block_size) :
                "scalar_fallback_policy";
        LLAMA_LOG_INFO("%s: fallback-report tensor=%-36s shape=[%" PRId64 ", %" PRId64 "] requested=%-7s base=%-7s reason=%s effective_candidates=%s selected=%-7s source=%s\n",
                __func__, tm.name.c_str(), ncols, nrows, ggml_type_name(requested_type), ggml_type_name(original_type),
                reason.c_str(), format_type_list(effective_candidates).c_str(), ggml_type_name(best_type),
                imatrix ? "imatrix-weighted" : "unweighted");

        LLAMA_LOG_INFO("%s: scalar-fallback tensor=%-36s shape=%" PRId64 "x%" PRId64 " original=%-7s selected=%-7s distortion=%10.6g bpw=%6.3f source=%s\n",
                __func__, tm.name.c_str(), nrows, ncols, ggml_type_name(original_type), ggml_type_name(best_type),
                best_distortion, best_bpw, imatrix ? "imatrix-weighted" : "unweighted");
    }

    if (evaluated > 0) {
        LLAMA_LOG_INFO("%s: shape-aware scalar fallback evaluated=%d changed=%d\n",
                __func__, evaluated, changed);
    }
}

static void apply_spqr_repair(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->quant_repair || !params->quant_repair_gain || params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }
    ensure_logit_gate_loaded(qs);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int evaluated = 0;
    int repaired = 0;
    int kept_selected = 0;
    const bool budget_limited = quant_budget_limited(params);
    const repair_gate_settings gate_settings = repair_gate_for_context(qs, budget_limited, false);
    const float cost_lambda = budget_limited ?
        std::max(params->rd_lambda, qs.rd_allocation_lambda) :
        params->rd_lambda;
    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const ggml_type selected_type = tm.target_type;
        const std::vector<ggml_type> candidates = spqr_repair_candidate_types(params, tensor, selected_type);
        if (candidates.empty()) {
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

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const float distortion_weight = rd_distortion_weight(tm);
        const spqr_repair_eval selected = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, selected_type, sample_rows, distortion_weight);
        if (selected.type == GGML_TYPE_COUNT) {
            continue;
        }
        const teacher_gate_eval selected_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, selected.type, sample_rows, selected.composite_error);
        const logit_teacher_risk_breakdown teacher_risk_parts = logit_local_teacher_risk_breakdown(qs, tm);
        const float teacher_risk = teacher_risk_parts.total;
        const float demotion_risk = logit_local_demotion_risk(teacher_risk_parts);
        const float promotion_pressure = logit_local_promotion_pressure(teacher_risk_parts);
        const float local_accept_ratio = logit_local_accept_ratio(gate_settings.accept_ratio, demotion_risk);
        const float local_max_error = logit_local_error_limit(gate_settings.max_error, demotion_risk);

        ++evaluated;
        spqr_repair_eval best = selected;
        teacher_gate_eval best_gate = selected_gate;
        bool accepted = false;
        std::vector<ggml_type> all_candidates = candidates;
        if (!budget_limited && promotion_pressure > 0.02f) {
            const std::vector<ggml_type> promote_candidates =
                spqr_repair_promotion_candidate_types(params, tensor, selected_type);
            all_candidates.insert(all_candidates.end(), promote_candidates.begin(), promote_candidates.end());
        }
        for (ggml_type candidate_type : all_candidates) {
            const spqr_repair_eval candidate = evaluate_spqr_repair_candidate(
                    tensor, data, imatrix, candidate_type, sample_rows, distortion_weight);
            if (candidate.type == GGML_TYPE_COUNT) {
                continue;
            }
            const teacher_gate_eval candidate_gate = evaluate_teacher_gate_error(
                    params, tensor, data, imatrix, candidate.type, sample_rows, candidate.composite_error);

            const bool is_promotion = candidate.bpw > selected.bpw + 1e-6f;
            bool choose_candidate = false;
            if (is_promotion) {
                const float promotion_lambda = cost_lambda / (1.0f + 2.5f * promotion_pressure);
                const float min_promotion_gain = std::max(0.0001f, selected_gate.gate_error * (0.01f + 0.04f * (1.0f - promotion_pressure)));
                const bool meaningful_promotion =
                    selected_gate.gate_error - candidate_gate.gate_error >= min_promotion_gain;
                const float candidate_cost = candidate_gate.gate_error + promotion_lambda * candidate.bpw;
                const float best_cost = best_gate.gate_error + promotion_lambda * best.bpw;
                choose_candidate = promotion_pressure > 0.0f && meaningful_promotion && candidate_cost < best_cost;
            } else {
                const bool under_relative_gate =
                    candidate_gate.gate_error <= selected_gate.gate_error * local_accept_ratio;
                const bool under_absolute_gate =
                    candidate.weighted_mse <= local_max_error ||
                    (candidate_gate.proxy_error >= 0.0f && candidate_gate.proxy_error <= local_max_error);
                const float candidate_cost = demotion_risk * candidate_gate.gate_error + cost_lambda * candidate.bpw;
                const float best_cost = demotion_risk * best_gate.gate_error + cost_lambda * best.bpw;
                choose_candidate = (under_relative_gate || under_absolute_gate) && candidate_cost < best_cost;
            }
            if (choose_candidate) {
                best = candidate;
                best_gate = candidate_gate;
                accepted = true;
            }

            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: quant-repair-candidate tensor=%-36s selected=%-7s type=%-7s mode=%-8s weighted_mse=%10.6g teacher_proxy=%10.6g block=%10.6g rank=%10.6g feature=%10.6g feature_l1=%10.6g feature_cos=%10.6g feature_norm=%10.6g gate_error=%10.6g gain=%9.6g shape=%9.6g outlier=%6.3f bpw=%6.3f selected_gate=%10.6g teacher_risk=%6.3f demotion_risk=%6.3f promotion_pressure=%5.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) local_accept_ratio=%.3f local_max_error=%10.6g accepted=%s\n",
                        __func__, tm.name.c_str(), ggml_type_name(selected.type), ggml_type_name(candidate.type), candidate.weighted_mse,
                        is_promotion ? "promote" : "demote",
                        candidate_gate.proxy_error,
                        candidate_gate.block_error,
                        candidate_gate.rank_error,
                        candidate_gate.feature_error,
                        candidate_gate.feature_l1_error,
                        candidate_gate.feature_cosine_error,
                        candidate_gate.feature_norm_error,
                        candidate_gate.gate_error,
                        candidate.gain_error, candidate.shape_error, candidate.outlier_concentration,
                        candidate.bpw, selected_gate.gate_error, teacher_risk, demotion_risk, promotion_pressure, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family, local_accept_ratio, local_max_error,
                        (candidate.type == best.type && accepted) ? "yes" : "no");
            }
        }

        if (accepted && best.type != selected_type) {
            apply_quant_transition(tm, best.type, best.weighted_mse, distortion_weight, best.bpw, best_gate.gate_error + cost_lambda * best.bpw);
            ++repaired;
            LLAMA_LOG_INFO("%s: quant-repair tensor=%-36s original=%-7s repaired=%-7s original_error=%10.6g repaired_error=%10.6g original_gate=%10.6g repaired_gate=%10.6g repaired_feature=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) gain=%9.6g shape=%9.6g outlier=%6.3f bpw=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(selected.type), ggml_type_name(best.type), selected.weighted_mse,
                    best.weighted_mse, selected_gate.gate_error, best_gate.gate_error, best_gate.feature_error, teacher_risk, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family,
                    best.gain_error, best.shape_error, best.outlier_concentration, best.bpw);
        } else {
            ++kept_selected;
            LLAMA_LOG_INFO("%s: quant-repair tensor=%-36s selected=%-7s reason=kept selected_error=%10.6g selected_gate=%10.6g selected_feature=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) gain=%9.6g shape=%9.6g outlier=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(selected.type),
                    selected.weighted_mse, selected_gate.gate_error, selected_gate.feature_error, teacher_risk, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family,
                    selected.gain_error, selected.shape_error, selected.outlier_concentration);
        }
    }

    if (evaluated > 0) {
        LLAMA_LOG_INFO("%s: quant-repair evaluated=%d repaired=%d kept=%d methods=gain teacher_aware=%s teacher_mix=%.2f block_mix=%.2f rank_mix=%.2f feature_mix=%.2f top_k=%d accept_ratio=%.3f configured_accept_ratio=%.3f max_error=%g configured_max_error=%g budget_limited=%s logit_gate=%s\n",
                __func__, evaluated, repaired, kept_selected,
                params->quant_teacher_aware ? "yes" : "no",
                params->quant_teacher_aware_mix,
                params->quant_teacher_aware_block_mix,
                params->quant_teacher_aware_rank_mix,
                teacher_feature_gate_mix(params),
                params->quant_teacher_aware_top_k,
                gate_settings.accept_ratio, params->quant_repair_accept_ratio,
                gate_settings.max_error, params->quant_repair_max_error,
                budget_limited ? "yes" : "no",
                (params->logit_gate && qs.logit_report_available) ? (qs.logit_gate_pass ? "pass" : "fail") : "off");
    }
}

static std::vector<compression_opportunity> collect_compression_opportunities(
        quantize_state_impl & qs,
        const llama_model_quantize_params * params,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        const std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread,
        bool best_per_tensor) {
    ensure_logit_gate_loaded(qs);
    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::vector<compression_opportunity> opportunities;
    const bool budget_limited = quant_budget_limited(params);
    const repair_gate_settings gate_settings = repair_gate_for_context(qs, budget_limited, false);
    const bool repair_probe_enabled = params->quant_repair &&
        (params->quant_repair_clipping || params->quant_repair_scale);
    const float effective_min_error = gate_settings.min_error;
    const float effective_min_improvement = gate_settings.min_improvement;
    const int max_layer = max_transformer_layer(metadata);

    for (size_t i = 0; i < metadata.size(); ++i) {
        const tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const ggml_type selected_type = tm.target_type;
        const std::vector<ggml_type> candidates = spqr_repair_candidate_types(params, tensor, selected_type);
        if (candidates.empty()) {
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

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const float distortion_weight = rd_distortion_weight(tm);
        const spqr_repair_eval selected = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, selected_type, sample_rows, distortion_weight);
        if (selected.type == GGML_TYPE_COUNT) {
            continue;
        }
        const teacher_gate_eval selected_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, selected.type, sample_rows, selected.composite_error);
        const logit_teacher_risk_breakdown teacher_risk_parts = logit_local_teacher_risk_breakdown(qs, tm);
        const float teacher_risk = teacher_risk_parts.total;
        const float safe_pressure = logit_local_safe_compression_pressure(teacher_risk_parts);
        const float promotion_pressure = logit_local_promotion_pressure(teacher_risk_parts);
        const float demotion_risk = logit_local_demotion_risk(teacher_risk_parts);
        const float local_accept_ratio = logit_local_accept_ratio(gate_settings.accept_ratio, demotion_risk);
        const float local_max_error = logit_local_error_limit(gate_settings.max_error, demotion_risk);

        compression_opportunity best;
        for (ggml_type candidate_type : candidates) {
            const spqr_repair_eval candidate = evaluate_spqr_repair_candidate(
                    tensor, data, imatrix, candidate_type, sample_rows, distortion_weight);
            if (candidate.type == GGML_TYPE_COUNT) {
                continue;
            }
            const teacher_gate_eval candidate_gate = evaluate_teacher_gate_error(
                    params, tensor, data, imatrix, candidate.type, sample_rows, candidate.composite_error);

            const size_t selected_bytes = estimate_quantized_tensor_bytes(tensor, selected.type);
            const size_t candidate_bytes = estimate_quantized_tensor_bytes(tensor, candidate.type);
            if (candidate_bytes >= selected_bytes) {
                continue;
            }

            const bool proxy_safe =
                candidate_gate.gate_error <= selected_gate.gate_error * local_accept_ratio ||
                candidate.weighted_mse <= local_max_error ||
                (candidate_gate.proxy_error >= 0.0f && candidate_gate.proxy_error <= local_max_error);
            float repaired_candidate_error = candidate_gate.gate_error;
            bool repair_potential = false;
            if (!proxy_safe && repair_probe_enabled) {
                const spqr_teacher_repair_result repair = evaluate_teacher_repair_clipping(
                        tensor, data, imatrix, candidate.type, sample_rows,
                        effective_min_error, effective_min_improvement,
                        params->quant_repair_clipping, params->quant_repair_scale);
                if (repair.base_error >= 0.0f) {
                    repaired_candidate_error = repair.repaired_error;
                    repair_potential =
                        repair.repaired_error <= selected_gate.gate_error * local_accept_ratio ||
                        repair.repaired_error <= local_max_error;
                }
            }

            const float effective_candidate_error = repair_potential ? repaired_candidate_error : candidate_gate.gate_error;
            const float delta_error = demotion_risk * std::max(0.0f, effective_candidate_error - selected_gate.gate_error);
            const float saved_mib = (float) (selected_bytes - candidate_bytes) / 1024.0f / 1024.0f;
            const float budget_bias = budget_limited ? budget_layer_position_bias(tm, max_layer) : 1.0f;
            const float score = budget_bias * (1.0f + 0.75f * safe_pressure) * saved_mib /
                std::max(delta_error * (1.0f + 0.50f * promotion_pressure), 1e-9f);

            compression_opportunity opportunity;
            opportunity.tensor_index = i;
            opportunity.name = tm.name;
            opportunity.selected_type = selected.type;
            opportunity.candidate_type = candidate.type;
            opportunity.saved_bytes = selected_bytes - candidate_bytes;
            opportunity.selected_error = selected.composite_error;
            opportunity.selected_gate_error = selected_gate.gate_error;
            opportunity.candidate_error = candidate.composite_error;
            opportunity.candidate_gate_error = candidate_gate.gate_error;
            opportunity.candidate_block_error = candidate_gate.block_error;
            opportunity.candidate_rank_error = candidate_gate.rank_error;
            opportunity.candidate_feature_error = candidate_gate.feature_error;
            opportunity.teacher_risk = teacher_risk;
            opportunity.teacher_risk_tensor = teacher_risk_parts.tensor;
            opportunity.teacher_risk_layer = teacher_risk_parts.layer;
            opportunity.teacher_risk_family = teacher_risk_parts.family;
            opportunity.teacher_safe_pressure = safe_pressure;
            opportunity.teacher_promotion_pressure = promotion_pressure;
            opportunity.budget_bias = budget_bias;
            opportunity.repaired_candidate_error = repaired_candidate_error;
            opportunity.effective_candidate_error = effective_candidate_error;
            opportunity.candidate_weighted_mse = candidate.weighted_mse;
            opportunity.delta_error = delta_error;
            opportunity.opportunity_score = score;
            opportunity.selected_bpw = selected.bpw;
            opportunity.candidate_bpw = candidate.bpw;
            opportunity.proxy_safe = proxy_safe;
            opportunity.repair_potential = repair_potential;

            if (!opportunity.proxy_safe && !opportunity.repair_potential) {
                continue;
            }

            if (!best_per_tensor) {
                opportunities.push_back(opportunity);
                continue;
            }

            if (best.candidate_type == GGML_TYPE_COUNT ||
                    compression_opportunity_better(opportunity, best)) {
                best = opportunity;
            }
        }

        if (best_per_tensor && best.candidate_type != GGML_TYPE_COUNT) {
            opportunities.push_back(best);
        }
    }

    return opportunities;
}

static void print_compression_opportunity_report(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        const std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->print_compression_opportunity_report ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }
    std::vector<compression_opportunity> opportunities = collect_compression_opportunities(
            qs, params, ml, tensors, metadata, imatrix_data, nthread, true);
    int proxy_safe_count = 0;
    int repair_potential_count = 0;
    size_t proxy_safe_saved_bytes = 0;
    size_t repair_potential_saved_bytes = 0;
    size_t total_saved_bytes = 0;
    const bool budget_limited = quant_budget_limited(params);

    std::sort(opportunities.begin(), opportunities.end(), compression_opportunity_better);
    for (const compression_opportunity & opportunity : opportunities) {
        total_saved_bytes += opportunity.saved_bytes;
        if (opportunity.proxy_safe) {
            ++proxy_safe_count;
            proxy_safe_saved_bytes += opportunity.saved_bytes;
        } else if (opportunity.repair_potential) {
            ++repair_potential_count;
            repair_potential_saved_bytes += opportunity.saved_bytes;
        }
    }

    const size_t report_count = std::min<size_t>(opportunities.size(), 32);
    LLAMA_LOG_INFO("%s: compression-opportunity summary evaluated=%d candidates=%d proxy_safe=%d proxy_safe_saved=%8.2f MiB repair_potential=%d repair_potential_saved=%8.2f MiB total_best_saved=%8.2f MiB teacher_aware=%s teacher_mix=%.2f block_mix=%.2f rank_mix=%.2f feature_mix=%.2f top_k=%d budget_limited=%s bottom_first_bias=%s\n",
            __func__, (int) opportunities.size(), (int) opportunities.size(), proxy_safe_count,
            proxy_safe_saved_bytes/1024.0/1024.0,
            repair_potential_count, repair_potential_saved_bytes/1024.0/1024.0,
            total_saved_bytes/1024.0/1024.0,
            params->quant_teacher_aware ? "yes" : "no",
            params->quant_teacher_aware_mix,
            params->quant_teacher_aware_block_mix,
            params->quant_teacher_aware_rank_mix,
            teacher_feature_gate_mix(params),
            params->quant_teacher_aware_top_k,
            budget_limited ? "yes" : "no",
            budget_limited ? "on" : "off");
    for (size_t rank = 0; rank < report_count; ++rank) {
        const compression_opportunity & opportunity = opportunities[rank];
        LLAMA_LOG_INFO("%s: compression-opportunity rank=%3d tensor=%-36s selected=%-7s candidate=%-7s saved=%8.2f MiB delta_error=%10.6g selected_error=%10.6g selected_gate=%10.6g candidate_error=%10.6g candidate_gate=%10.6g block=%10.6g rank=%10.6g feature=%10.6g repaired_candidate_error=%10.6g score=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) safe_pressure=%5.3f promotion_pressure=%5.3f budget_bias=%5.3f bpw=%6.3f->%6.3f proxy_safe=%s repair_potential=%s\n",
                __func__, (int) rank + 1, opportunity.name.c_str(),
                ggml_type_name(opportunity.selected_type), ggml_type_name(opportunity.candidate_type),
                opportunity.saved_bytes/1024.0/1024.0,
                opportunity.delta_error,
                opportunity.selected_error,
                opportunity.selected_gate_error,
                opportunity.candidate_error,
                opportunity.candidate_gate_error,
                opportunity.candidate_block_error,
                opportunity.candidate_rank_error,
                opportunity.candidate_feature_error,
                opportunity.repaired_candidate_error,
                opportunity.opportunity_score,
                opportunity.teacher_risk,
                opportunity.teacher_risk_tensor,
                opportunity.teacher_risk_layer,
                opportunity.teacher_risk_family,
                opportunity.teacher_safe_pressure,
                opportunity.teacher_promotion_pressure,
                opportunity.budget_bias,
                opportunity.selected_bpw,
                opportunity.candidate_bpw,
                opportunity.proxy_safe ? "yes" : "no",
                opportunity.repair_potential ? "yes" : "no");
    }
}

static std::vector<promotion_opportunity> collect_promotion_opportunities(
        quantize_state_impl & qs,
        const llama_model_quantize_params * params,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        const std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    std::vector<promotion_opportunity> opportunities;
    if (!params->logit_guided_search || !params->quant_teacher_aware ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return opportunities;
    }

    ensure_logit_gate_loaded(qs);
    if (!qs.logit_report_available || !qs.logit_report_paired) {
        return opportunities;
    }

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int max_layer = 0;
    for (const tensor_metadata & tm : metadata) {
        max_layer = std::max(max_layer, tm.layer);
    }

    for (size_t i = 0; i < metadata.size(); ++i) {
        const tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const logit_teacher_risk_breakdown teacher_risk_parts = logit_local_teacher_risk_breakdown(qs, tm);
        const float promotion_pressure = logit_local_promotion_pressure(teacher_risk_parts);
        if (promotion_pressure <= 0.02f) {
            continue;
        }

        const ggml_type selected_type = tm.target_type;
        const std::vector<ggml_type> candidates =
            spqr_repair_promotion_candidate_types(params, tensor, selected_type);
        if (candidates.empty()) {
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

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const float distortion_weight = rd_distortion_weight(tm);
        const spqr_repair_eval selected = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, selected_type, sample_rows, distortion_weight);
        if (selected.type == GGML_TYPE_COUNT) {
            continue;
        }
        const teacher_gate_eval selected_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, selected.type, sample_rows, selected.composite_error);

        promotion_opportunity best;
        for (ggml_type candidate_type : candidates) {
            const spqr_repair_eval candidate = evaluate_spqr_repair_candidate(
                    tensor, data, imatrix, candidate_type, sample_rows, distortion_weight);
            if (candidate.type == GGML_TYPE_COUNT) {
                continue;
            }
            const size_t selected_bytes = estimate_quantized_tensor_bytes(tensor, selected.type);
            const size_t candidate_bytes = estimate_quantized_tensor_bytes(tensor, candidate.type);
            if (candidate_bytes <= selected_bytes) {
                continue;
            }

            const teacher_gate_eval candidate_gate = evaluate_teacher_gate_error(
                    params, tensor, data, imatrix, candidate.type, sample_rows, candidate.composite_error);
            const float raw_damage_reduction = std::max(0.0f, selected_gate.gate_error - candidate_gate.gate_error);
            if (raw_damage_reduction <= 0.0f) {
                continue;
            }

            const float min_promotion_gain = std::max(
                    0.00005f,
                    selected_gate.gate_error * (0.01f + 0.03f * (1.0f - promotion_pressure)));
            if (raw_damage_reduction < min_promotion_gain) {
                continue;
            }

            const float safe_pressure = logit_local_safe_compression_pressure(teacher_risk_parts);
            const float quality_bias = quality_layer_position_bias(tm, max_layer);
            const float effective_damage_reduction =
                raw_damage_reduction *
                (1.0f + 1.25f * promotion_pressure) *
                (1.0f - 0.20f * safe_pressure) *
                quality_bias;
            const float extra_mib = (float) (candidate_bytes - selected_bytes) / 1024.0f / 1024.0f;
            const float score = effective_damage_reduction / std::max(extra_mib, 1e-9f);

            promotion_opportunity opportunity;
            opportunity.tensor_index = i;
            opportunity.name = tm.name;
            opportunity.selected_type = selected.type;
            opportunity.candidate_type = candidate.type;
            opportunity.extra_bytes = candidate_bytes - selected_bytes;
            opportunity.selected_error = selected.composite_error;
            opportunity.selected_gate_error = selected_gate.gate_error;
            opportunity.candidate_error = candidate.composite_error;
            opportunity.candidate_gate_error = candidate_gate.gate_error;
            opportunity.candidate_block_error = candidate_gate.block_error;
            opportunity.candidate_rank_error = candidate_gate.rank_error;
            opportunity.candidate_feature_error = candidate_gate.feature_error;
            opportunity.candidate_weighted_mse = candidate.weighted_mse;
            opportunity.teacher_risk = teacher_risk_parts.total;
            opportunity.teacher_risk_tensor = teacher_risk_parts.tensor;
            opportunity.teacher_risk_layer = teacher_risk_parts.layer;
            opportunity.teacher_risk_family = teacher_risk_parts.family;
            opportunity.teacher_safe_pressure = safe_pressure;
            opportunity.teacher_promotion_pressure = promotion_pressure;
            opportunity.quality_bias = quality_bias;
            opportunity.damage_reduction = effective_damage_reduction;
            opportunity.opportunity_score = score;
            opportunity.selected_bpw = selected.bpw;
            opportunity.candidate_bpw = candidate.bpw;

            if (best.candidate_type == GGML_TYPE_COUNT || promotion_opportunity_better(opportunity, best)) {
                best = opportunity;
            }
        }

        if (best.candidate_type != GGML_TYPE_COUNT) {
            opportunities.push_back(best);
        }
    }

    return opportunities;
}

static promotion_package build_best_promotion_package(
        const llama_model_quantize_params * params,
        const promotion_opportunity & promotion,
        const std::vector<compression_opportunity> & financing_pool,
        const std::vector<tensor_metadata> & metadata,
        size_t headroom_bytes) {
    promotion_package best;
    const size_t required_financing_bytes =
        promotion.extra_bytes > headroom_bytes ? (promotion.extra_bytes - headroom_bytes) : 0;

    if (required_financing_bytes == 0) {
        evaluate_promotion_package(params, promotion, {}, metadata, required_financing_bytes, best);
        return best;
    }

    std::vector<compression_opportunity> eligible;
    eligible.reserve(financing_pool.size());
    for (const compression_opportunity & demotion : financing_pool) {
        if (demotion.teacher_promotion_pressure > 0.05f) {
            continue;
        }
        if (demotion.teacher_risk > 1.05f && !demotion.proxy_safe) {
            continue;
        }
        if (demotion_package_conflicts_with_promotion(demotion, promotion, metadata)) {
            continue;
        }
        eligible.push_back(demotion);
        if ((int) eligible.size() >= std::max(1, params->logit_search_package_pool_size)) {
            break;
        }
    }

    promotion_package candidate;
    for (size_t i = 0; i < eligible.size(); ++i) {
        candidate = {};
        evaluate_promotion_package(params, promotion, { eligible[i] }, metadata, required_financing_bytes, candidate);
        if (promotion_package_better(candidate, best)) {
            best = candidate;
        }
    }

    if (params->logit_search_package_max_items >= 2) {
        for (size_t i = 0; i < eligible.size(); ++i) {
            for (size_t j = i + 1; j < eligible.size(); ++j) {
                if (eligible[i].tensor_index == eligible[j].tensor_index) {
                    continue;
                }
                candidate = {};
                evaluate_promotion_package(params, promotion, { eligible[i], eligible[j] }, metadata, required_financing_bytes, candidate);
                if (promotion_package_better(candidate, best)) {
                    best = candidate;
                }
            }
        }
    }

    if (params->logit_search_package_max_items >= 3) {
        for (size_t i = 0; i < eligible.size(); ++i) {
            for (size_t j = i + 1; j < eligible.size(); ++j) {
                if (eligible[i].tensor_index == eligible[j].tensor_index) {
                    continue;
                }
                for (size_t k = j + 1; k < eligible.size(); ++k) {
                    if (eligible[k].tensor_index == eligible[i].tensor_index ||
                            eligible[k].tensor_index == eligible[j].tensor_index) {
                        continue;
                    }
                    candidate = {};
                    evaluate_promotion_package(
                            params, promotion, { eligible[i], eligible[j], eligible[k] },
                            metadata, required_financing_bytes, candidate);
                    if (promotion_package_better(candidate, best)) {
                        best = candidate;
                    }
                }
            }
        }
    }

    return best;
}

static void apply_budget_repair_shrink(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->quant_repair || !quant_budget_limited(params) ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }
    ensure_logit_gate_loaded(qs);

    const quant_budget_state budget = quant_budget_snapshot(qs, params, ml, tensors, metadata);
    const size_t target_bytes = budget.target_bytes;
    size_t current_bytes = budget.current_bytes;
    if (current_bytes <= target_bytes) {
        if (params->print_rd_report) {
            LLAMA_LOG_INFO("%s: skipped current=%8.2f MiB target=%8.2f MiB reason=already_within_budget\n",
                    __func__, current_bytes/1024.0/1024.0, target_bytes/1024.0/1024.0);
        }
        return;
    }

    const float cost_lambda = std::max(params->rd_lambda, qs.rd_allocation_lambda);
    const int max_passes = 4;
    int passes = 0;
    int demoted = 0;
    int proxy_safe_demotions = 0;
    int repair_potential_demotions = 0;
    float added_quality_cost = 0.0f;
    size_t total_saved_bytes = 0;
    int auction_candidates = 0;
    int auction_selected = 0;

    while (current_bytes > target_bytes && passes < max_passes) {
        std::vector<compression_opportunity> opportunities = collect_compression_opportunities(
                qs, params, ml, tensors, metadata, imatrix_data, nthread, false);
        if (opportunities.empty()) {
            break;
        }

        auction_candidates += (int) opportunities.size();
        std::sort(opportunities.begin(), opportunities.end(), compression_opportunity_better);

        bool applied_in_pass = false;
        std::vector<bool> touched(metadata.size(), false);
        for (size_t rank = 0; rank < opportunities.size(); ++rank) {
            const compression_opportunity & opportunity = opportunities[rank];
            if (current_bytes <= target_bytes) {
                break;
            }
            if (touched[opportunity.tensor_index]) {
                continue;
            }

            tensor_metadata & tm = metadata[opportunity.tensor_index];
            if (tm.target_type != opportunity.selected_type) {
                continue;
            }

            apply_quant_transition(
                    tm,
                    opportunity.candidate_type,
                    opportunity.candidate_weighted_mse,
                    rd_distortion_weight(tm),
                    opportunity.candidate_bpw,
                    opportunity.effective_candidate_error + cost_lambda * opportunity.budget_bias * opportunity.candidate_bpw);

            current_bytes -= opportunity.saved_bytes;
            total_saved_bytes += opportunity.saved_bytes;
            added_quality_cost += opportunity.delta_error;
            ++demoted;
            ++auction_selected;
            proxy_safe_demotions += opportunity.proxy_safe ? 1 : 0;
            repair_potential_demotions += opportunity.repair_potential ? 1 : 0;
            touched[opportunity.tensor_index] = true;
            applied_in_pass = true;

            LLAMA_LOG_INFO("%s: shrink-pass=%d auction-rank=%3d tensor=%-36s selected=%-7s demoted=%-7s saved=%8.2f MiB quality_delta=%10.6g value=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) safe_pressure=%5.3f promotion_pressure=%5.3f budget_bias=%5.3f mode=%s current=%8.2f MiB target=%8.2f MiB\n",
                    __func__, passes + 1, (int) rank + 1, tm.name.c_str(),
                    ggml_type_name(opportunity.selected_type), ggml_type_name(opportunity.candidate_type),
                    opportunity.saved_bytes/1024.0/1024.0,
                    opportunity.delta_error,
                    opportunity.opportunity_score,
                    opportunity.teacher_risk,
                    opportunity.teacher_risk_tensor,
                    opportunity.teacher_risk_layer,
                    opportunity.teacher_risk_family,
                    opportunity.teacher_safe_pressure,
                    opportunity.teacher_promotion_pressure,
                    opportunity.budget_bias,
                    opportunity.proxy_safe ? "proxy-safe" : "repair-potential",
                    current_bytes/1024.0/1024.0,
                    target_bytes/1024.0/1024.0);
        }

        if (!applied_in_pass) {
            break;
        }
        ++passes;
    }

    if (demoted > 0 || current_bytes > target_bytes) {
        LLAMA_LOG_INFO("%s: summary mode=auction demoted=%d proxy_safe=%d repair_potential=%d candidates=%d selected=%d saved=%8.2f MiB added_quality_cost=%10.6g cost_per_mib=%10.6g actual=%8.2f MiB target=%8.2f MiB difference=%+.2f MiB passes=%d status=%s bottom_first_bias=on bias_range=0.94..1.08\n",
                __func__,
                demoted,
                proxy_safe_demotions,
                repair_potential_demotions,
                auction_candidates,
                auction_selected,
                total_saved_bytes/1024.0/1024.0,
                added_quality_cost,
                total_saved_bytes > 0 ? added_quality_cost / ((float) total_saved_bytes/1024.0f/1024.0f) : 0.0f,
                current_bytes/1024.0/1024.0,
                target_bytes/1024.0/1024.0,
                ((double) current_bytes - target_bytes)/1024.0/1024.0,
                passes,
                current_bytes <= target_bytes ? "target-met" : "still-above-target");
    }
}

static void apply_logit_guided_budget_buyback(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->logit_guided_search || !quant_budget_limited(params) ||
            !params->quant_repair || !params->quant_teacher_aware ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }

    ensure_logit_gate_loaded(qs);
    if (!qs.logit_report_available || !qs.logit_report_paired) {
        if (params->print_rd_report) {
            LLAMA_LOG_INFO("%s: skipped reason=no_paired_logit_report\n", __func__);
        }
        return;
    }

    const quant_budget_state budget = quant_budget_snapshot(qs, params, ml, tensors, metadata);
    const size_t target_bytes = budget.target_bytes;
    const size_t extra_budget_bytes = (size_t) std::llround(
            std::max(0.0f, params->logit_search_promote_budget_mib) * 1024.0f * 1024.0f);
    size_t current_bytes = budget.current_bytes;

    const size_t max_bytes = target_bytes + extra_budget_bytes;
    const float cost_lambda = std::max(params->rd_lambda, qs.rd_allocation_lambda);
    if (current_bytes >= max_bytes) {
        if (params->print_rd_report) {
            LLAMA_LOG_INFO("%s: skipped current=%8.2f MiB target=%8.2f MiB ceiling=%8.2f MiB reason=no_buyback_budget_left\n",
                    __func__, current_bytes/1024.0/1024.0, target_bytes/1024.0/1024.0, max_bytes/1024.0/1024.0);
        }
        return;
    }

    const int max_passes = 2;
    int passes = 0;
    int promoted = 0;
    int financed_demotions = 0;
    int auction_candidates = 0;
    int auction_selected = 0;
    float recovered_damage = 0.0f;
    float financing_cost = 0.0f;
    size_t spent_bytes = 0;
    size_t financed_saved_bytes = 0;

    while (current_bytes < max_bytes && passes < max_passes) {
        std::vector<promotion_opportunity> opportunities = collect_promotion_opportunities(
                qs, params, ml, tensors, metadata, imatrix_data, nthread);
        if (opportunities.empty()) {
            break;
        }
        std::vector<compression_opportunity> financing_opportunities = collect_compression_opportunities(
                qs, params, ml, tensors, metadata, imatrix_data, nthread, false);
        std::sort(financing_opportunities.begin(), financing_opportunities.end(), compression_opportunity_better);
        std::sort(opportunities.begin(), opportunities.end(), promotion_opportunity_better);
        const size_t consider_count = std::min<size_t>(opportunities.size(), std::max(1, params->logit_search_top_k));
        auction_candidates += (int) consider_count;

        const size_t headroom_bytes = current_bytes < max_bytes ? (max_bytes - current_bytes) : 0;
        std::vector<promotion_package> packages;
        packages.reserve(consider_count);
        for (size_t rank = 0; rank < consider_count; ++rank) {
            const promotion_opportunity & opportunity = opportunities[rank];
            promotion_package best_package = build_best_promotion_package(
                    params, opportunity, financing_opportunities, metadata, headroom_bytes);
            if (!best_package.valid && params->print_rd_report) {
                LLAMA_LOG_INFO("%s: package-rejected rank=%3d tensor=%-36s selected=%-7s promoted=%-7s required=%8.2f MiB best_saved=%8.2f MiB package_damage=%10.6g overshoot=%7.3f correlation_penalty=%10.6g uncertainty_penalty=%10.6g net_gain=%10.6g reason=no_package_with_positive_net_value\n",
                        __func__, (int) rank + 1, opportunity.name.c_str(),
                        ggml_type_name(opportunity.selected_type), ggml_type_name(opportunity.candidate_type),
                        opportunity.extra_bytes/1024.0/1024.0,
                        best_package.saved_bytes/1024.0/1024.0,
                        best_package.package_damage,
                        best_package.overshoot_mib,
                        best_package.correlation_penalty,
                        best_package.uncertainty_penalty,
                        best_package.net_gain);
            }
            packages.push_back(best_package);
        }
        std::sort(packages.begin(), packages.end(), promotion_package_better);

        bool applied_in_pass = false;
        std::vector<bool> touched(metadata.size(), false);
        for (size_t rank = 0; rank < packages.size(); ++rank) {
            const promotion_package & package = packages[rank];
            if (!package.valid) {
                continue;
            }
            const promotion_opportunity & opportunity = package.promotion;
            if (touched[opportunity.tensor_index]) {
                continue;
            }
            if (metadata[opportunity.tensor_index].target_type != opportunity.selected_type) {
                continue;
            }

            bool package_conflict = false;
            size_t package_saved_bytes = 0;
            float package_financing_cost = 0.0f;
            for (const compression_opportunity & demotion : package.demotions) {
                if (touched[demotion.tensor_index] ||
                        metadata[demotion.tensor_index].target_type != demotion.selected_type) {
                    package_conflict = true;
                    break;
                }
                package_saved_bytes += demotion.saved_bytes;
                package_financing_cost += demotion.delta_error;
            }
            if (package_conflict) {
                continue;
            }

            const size_t dynamic_headroom_bytes = current_bytes < max_bytes ? (max_bytes - current_bytes) : 0;
            const size_t required_financing_bytes =
                opportunity.extra_bytes > dynamic_headroom_bytes ? (opportunity.extra_bytes - dynamic_headroom_bytes) : 0;
            if (package_saved_bytes < required_financing_bytes) {
                continue;
            }

            for (const compression_opportunity & demotion : package.demotions) {
                tensor_metadata & funding_tm = metadata[demotion.tensor_index];
                apply_quant_transition(
                        funding_tm,
                        demotion.candidate_type,
                        demotion.candidate_weighted_mse,
                        rd_distortion_weight(funding_tm),
                        demotion.candidate_bpw,
                        demotion.effective_candidate_error + cost_lambda * demotion.budget_bias * demotion.candidate_bpw);

                current_bytes -= demotion.saved_bytes;
                financed_saved_bytes += demotion.saved_bytes;
                financing_cost += demotion.delta_error;
                ++financed_demotions;
                touched[demotion.tensor_index] = true;
            }

            if (current_bytes + opportunity.extra_bytes > max_bytes) {
                for (const compression_opportunity & demotion : package.demotions) {
                    touched[demotion.tensor_index] = false;
                }
                break;
            }

            tensor_metadata & tm = metadata[opportunity.tensor_index];
            apply_quant_transition(
                    tm,
                    opportunity.candidate_type,
                    opportunity.candidate_weighted_mse,
                    rd_distortion_weight(tm),
                    opportunity.candidate_bpw,
                    opportunity.candidate_gate_error + cost_lambda * opportunity.candidate_bpw);

            current_bytes += opportunity.extra_bytes;
            spent_bytes += opportunity.extra_bytes;
            recovered_damage += opportunity.damage_reduction;
            ++promoted;
            ++auction_selected;
            touched[opportunity.tensor_index] = true;
            applied_in_pass = true;

            LLAMA_LOG_INFO("%s: buyback-pass=%d auction-rank=%3d tensor=%-36s selected=%-7s promoted=%-7s spent=%8.2f MiB financed_by=%d financed_saved=%8.2f MiB finance_cost=%10.6g net_gain=%10.6g overshoot=%7.3f correlation_penalty=%10.6g uncertainty_penalty=%10.6g damage_reduction=%10.6g value=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) safe_pressure=%5.3f promotion_pressure=%5.3f quality_bias=%5.3f current=%8.2f MiB target=%8.2f MiB ceiling=%8.2f MiB\n",
                    __func__, passes + 1, (int) rank + 1, tm.name.c_str(),
                    ggml_type_name(opportunity.selected_type), ggml_type_name(opportunity.candidate_type),
                    opportunity.extra_bytes/1024.0/1024.0,
                    (int) package.demotions.size(),
                    package_saved_bytes/1024.0/1024.0,
                    package_financing_cost,
                    package.net_gain,
                    package.overshoot_mib,
                    package.correlation_penalty,
                    package.uncertainty_penalty,
                    opportunity.damage_reduction,
                    package.package_score,
                    opportunity.teacher_risk,
                    opportunity.teacher_risk_tensor,
                    opportunity.teacher_risk_layer,
                    opportunity.teacher_risk_family,
                    opportunity.teacher_safe_pressure,
                    opportunity.teacher_promotion_pressure,
                    opportunity.quality_bias,
                    current_bytes/1024.0/1024.0,
                    target_bytes/1024.0/1024.0,
                    max_bytes/1024.0/1024.0);
            for (size_t i = 0; i < package.demotions.size(); ++i) {
                const compression_opportunity & demotion = package.demotions[i];
                LLAMA_LOG_INFO("%s: buyback-funding pass=%d rank=%3d item=%d tensor=%-36s selected=%-7s demoted=%-7s saved=%8.2f MiB damage=%10.6g safe_pressure=%5.3f promotion_pressure=%5.3f budget_bias=%5.3f\n",
                        __func__, passes + 1, (int) rank + 1, (int) i + 1, demotion.name.c_str(),
                        ggml_type_name(demotion.selected_type), ggml_type_name(demotion.candidate_type),
                        demotion.saved_bytes/1024.0/1024.0,
                        demotion.delta_error,
                        demotion.teacher_safe_pressure,
                        demotion.teacher_promotion_pressure,
                        demotion.budget_bias);
            }
        }

        if (!applied_in_pass) {
            break;
        }
        ++passes;
    }

    if (promoted > 0 || params->print_rd_report) {
        LLAMA_LOG_INFO("%s: summary mode=buyback promoted=%d financed_demotions=%d candidates=%d selected=%d spent=%8.2f MiB financed_saved=%8.2f MiB financing_cost=%10.6g recovered_damage=%10.6g net_gain=%10.6g damage_per_mib=%10.6g actual=%8.2f MiB target=%8.2f MiB ceiling=%8.2f MiB difference_vs_target=%+.2f MiB passes=%d status=%s top_k=%d\n",
                __func__,
                promoted,
                financed_demotions,
                auction_candidates,
                auction_selected,
                spent_bytes/1024.0/1024.0,
                financed_saved_bytes/1024.0/1024.0,
                financing_cost,
                recovered_damage,
                recovered_damage - financing_cost,
                spent_bytes > 0 ? (recovered_damage - financing_cost) / ((float) spent_bytes/1024.0f/1024.0f) : 0.0f,
                current_bytes/1024.0/1024.0,
                target_bytes/1024.0/1024.0,
                max_bytes/1024.0/1024.0,
                ((double) current_bytes - target_bytes)/1024.0/1024.0,
                passes,
                promoted > 0 ? "buyback-applied" : "no-opportunities",
                params->logit_search_top_k);
    }
}

static void apply_quality_precision_validation(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->quant_repair || !params->quant_teacher_aware || quant_budget_limited(params) ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }
    ensure_logit_gate_loaded(qs);

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    float accept_ratio = std::max(params->quant_repair_accept_ratio, 1.15f);
    float max_delta = std::max(params->quant_repair_max_error, 0.0015f);
    const float max_damage_per_mib = 0.00025f;
    const float max_total_delta = 0.010f;
    if (params->logit_gate && qs.logit_report_available && !qs.logit_gate_pass) {
        accept_ratio = std::min(accept_ratio, 1.02f);
        max_delta = std::min(max_delta, 0.00075f);
    }

    int evaluated = 0;
    int demoted = 0;
    size_t saved_bytes = 0;
    float added_teacher_damage = 0.0f;

    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const ggml_type selected_type = tm.target_type;
        const std::vector<ggml_type> candidates = spqr_repair_candidate_types(params, tensor, selected_type);
        if (candidates.empty()) {
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

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const float distortion_weight = rd_distortion_weight(tm);
        const spqr_repair_eval selected = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, selected_type, sample_rows, distortion_weight);
        if (selected.type == GGML_TYPE_COUNT) {
            continue;
        }
        const teacher_gate_eval selected_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, selected.type, sample_rows, selected.composite_error);
        const logit_teacher_risk_breakdown teacher_risk_parts = logit_local_teacher_risk_breakdown(qs, tm);
        const float teacher_risk = teacher_risk_parts.total;
        const float demotion_risk = logit_local_demotion_risk(teacher_risk_parts);
        const float local_accept_ratio = logit_local_accept_ratio(accept_ratio, demotion_risk);
        const float local_max_delta = max_delta / std::max(0.80f, demotion_risk);
        const float local_max_total_delta = max_total_delta / std::max(0.80f, demotion_risk);
        const float local_max_damage_per_mib = max_damage_per_mib / std::max(0.80f, demotion_risk);

        ++evaluated;
        ggml_type best_type = GGML_TYPE_COUNT;
        spqr_repair_eval best_eval;
        teacher_gate_eval best_gate;
        size_t best_saved_bytes = 0;
        float best_delta = 0.0f;
        float best_score = -1.0f;

        const size_t selected_bytes = estimate_quantized_tensor_bytes(tensor, selected.type);
        for (ggml_type candidate_type : candidates) {
            const spqr_repair_eval candidate = evaluate_spqr_repair_candidate(
                    tensor, data, imatrix, candidate_type, sample_rows, distortion_weight);
            if (candidate.type == GGML_TYPE_COUNT) {
                continue;
            }

            const size_t candidate_bytes = estimate_quantized_tensor_bytes(tensor, candidate.type);
            if (candidate_bytes >= selected_bytes) {
                continue;
            }

            const teacher_gate_eval candidate_gate = evaluate_teacher_gate_error(
                    params, tensor, data, imatrix, candidate.type, sample_rows, candidate.composite_error);
            const float raw_delta = std::max(0.0f, candidate_gate.gate_error - selected_gate.gate_error);
            const float delta = demotion_risk * raw_delta;
            const float saved_mib = (float) (selected_bytes - candidate_bytes) / 1024.0f / 1024.0f;
            const float damage_per_mib = delta / std::max(saved_mib, 1e-9f);
            const bool low_relative_damage = candidate_gate.gate_error <= selected_gate.gate_error * local_accept_ratio;
            const bool low_absolute_damage = delta <= local_max_delta;
            const bool good_value = damage_per_mib <= local_max_damage_per_mib;
            const bool bounded_damage = delta <= local_max_total_delta;

            if (!(bounded_damage && (low_relative_damage || low_absolute_damage || good_value))) {
                continue;
            }

            const float score = saved_mib / std::max(delta, 1e-9f);
            if (score > best_score) {
                best_type = candidate.type;
                best_eval = candidate;
                best_gate = candidate_gate;
                best_saved_bytes = selected_bytes - candidate_bytes;
                best_delta = delta;
                best_score = score;
            }
        }

        if (best_type == GGML_TYPE_COUNT) {
            continue;
        }

        apply_quant_transition(
                tm,
                best_type,
                best_eval.weighted_mse,
                distortion_weight,
                best_eval.bpw,
                best_gate.gate_error + params->rd_lambda * best_eval.bpw);

        ++demoted;
        saved_bytes += best_saved_bytes;
        added_teacher_damage += best_delta;
        LLAMA_LOG_INFO("%s: precision-validation tensor=%-36s selected=%-7s demoted=%-7s saved=%8.2f MiB teacher_delta=%10.6g feature=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) score=%10.6g\n",
                __func__, tm.name.c_str(), ggml_type_name(selected.type), ggml_type_name(best_type),
                best_saved_bytes/1024.0/1024.0, best_delta, best_gate.feature_error, teacher_risk, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family, best_score);
    }

    if (evaluated > 0) {
        LLAMA_LOG_INFO("%s: precision-validation evaluated=%d demoted=%d saved=%8.2f MiB added_teacher_damage=%10.6g accept_ratio=%.3f max_delta=%g max_damage_per_mib=%g feature_mix=%.2f logit_gate=%s\n",
                __func__, evaluated, demoted, saved_bytes/1024.0/1024.0, added_teacher_damage,
                accept_ratio, max_delta, max_damage_per_mib, teacher_feature_gate_mix(params),
                (params->logit_gate && qs.logit_report_available) ? (qs.logit_gate_pass ? "pass" : "fail") : "off");
    }
}

static void apply_budget_first_type_cap(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    if (!params->quant_repair || !quant_budget_limited(params) ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int evaluated = 0;
    int capped = 0;
    int kept = 0;
    ensure_logit_gate_loaded(qs);
    const repair_gate_settings gate_settings = repair_gate_for_context(qs, true, false);
    const float cost_lambda = std::max(params->rd_lambda, qs.rd_allocation_lambda);
    const int max_layer = max_transformer_layer(metadata);

    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const ggml_type selected_type = tm.target_type;
        const std::vector<ggml_type> candidates = spqr_repair_candidate_types(params, tensor, selected_type);
        if (candidates.empty()) {
            continue;
        }

        const ggml_type capped_type = candidates.front();
        const float * imatrix = nullptr;
        if (imatrix_data) {
            auto it = imatrix_data->find(tm.remapped_imatrix_name);
            if (it != imatrix_data->end() && it->second.size() == (size_t) ncols * tensor->ne[2]) {
                imatrix = it->second.data();
            }
        }

        load_tensor_as_f32(ml, tensor, read_data, f32_conv, data, workers, nthread);

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const float distortion_weight = rd_distortion_weight(tm);
        const spqr_repair_eval selected = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, selected_type, sample_rows, distortion_weight);
        const spqr_repair_eval candidate = evaluate_spqr_repair_candidate(
                tensor, data, imatrix, capped_type, sample_rows, distortion_weight);
        if (selected.type == GGML_TYPE_COUNT || candidate.type == GGML_TYPE_COUNT) {
            continue;
        }
        const teacher_gate_eval selected_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, selected.type, sample_rows, selected.composite_error);
        const teacher_gate_eval candidate_gate = evaluate_teacher_gate_error(
                params, tensor, data, imatrix, candidate.type, sample_rows, candidate.composite_error);
        const logit_teacher_risk_breakdown teacher_risk_parts = logit_local_teacher_risk_breakdown(qs, tm);
        const float teacher_risk = teacher_risk_parts.total;
        const float demotion_risk = logit_local_demotion_risk(teacher_risk_parts);
        const float local_accept_ratio = logit_local_accept_ratio(gate_settings.accept_ratio, demotion_risk);
        const float local_max_error = logit_local_error_limit(gate_settings.max_error, demotion_risk);

        ++evaluated;
        const bool under_relative_gate =
            candidate_gate.gate_error <= selected_gate.gate_error * local_accept_ratio;
        const bool under_absolute_gate =
            candidate.weighted_mse <= local_max_error ||
            (candidate_gate.proxy_error >= 0.0f && candidate_gate.proxy_error <= local_max_error);
        const float budget_bias = budget_layer_position_bias(tm, max_layer);
        const float selected_cost = demotion_risk * selected_gate.gate_error + cost_lambda * budget_bias * selected.bpw;
        const float candidate_cost = demotion_risk * candidate_gate.gate_error + cost_lambda * budget_bias * candidate.bpw;
        if ((under_relative_gate || under_absolute_gate) && candidate_cost < selected_cost) {
            apply_quant_transition(
                    tm,
                    candidate.type,
                    candidate.weighted_mse,
                    distortion_weight,
                    candidate.bpw,
                    candidate_gate.gate_error + cost_lambda * budget_bias * candidate.bpw);
            ++capped;
            LLAMA_LOG_INFO("%s: budget-first tensor=%-36s original=%-7s capped=%-7s original_error=%10.6g capped_error=%10.6g original_gate=%10.6g capped_gate=%10.6g capped_feature=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) budget_bias=%5.3f original_bpw=%6.3f capped_bpw=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(selected.type), ggml_type_name(candidate.type),
                    selected.weighted_mse, candidate.weighted_mse,
                    selected_gate.gate_error, candidate_gate.gate_error, candidate_gate.feature_error, teacher_risk, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family, budget_bias,
                    selected.bpw, candidate.bpw);
        } else {
            ++kept;
            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: budget-first tensor=%-36s selected=%-7s capped=%-7s reason=kept selected_error=%10.6g capped_error=%10.6g selected_gate=%10.6g capped_gate=%10.6g capped_feature=%10.6g teacher_risk=%6.3f risk_parts=(t=%+.3f l=%+.3f f=%+.3f) budget_bias=%5.3f selected_bpw=%6.3f capped_bpw=%6.3f\n",
                        __func__, tm.name.c_str(), ggml_type_name(selected.type), ggml_type_name(candidate.type),
                        selected.weighted_mse, candidate.weighted_mse,
                        selected_gate.gate_error, candidate_gate.gate_error, candidate_gate.feature_error, teacher_risk, teacher_risk_parts.tensor, teacher_risk_parts.layer, teacher_risk_parts.family, budget_bias,
                        selected.bpw, candidate.bpw);
            }
        }
    }

    if (evaluated > 0) {
        LLAMA_LOG_INFO("%s: budget-first evaluated=%d capped=%d kept=%d teacher_aware=%s teacher_mix=%.2f block_mix=%.2f rank_mix=%.2f feature_mix=%.2f top_k=%d accept_ratio=%.3f max_error=%g cost_lambda=%.6g bottom_first_bias=on bias_range=0.94..1.08 logit_gate=%s\n",
                __func__, evaluated, capped, kept,
                params->quant_teacher_aware ? "yes" : "no",
                params->quant_teacher_aware_mix,
                params->quant_teacher_aware_block_mix,
                params->quant_teacher_aware_rank_mix,
                teacher_feature_gate_mix(params),
                params->quant_teacher_aware_top_k,
                gate_settings.accept_ratio, gate_settings.max_error, cost_lambda,
                (params->logit_gate && qs.logit_report_available) ? (qs.logit_gate_pass ? "pass" : "fail") : "off");
    }
}

static void apply_spqr_teacher_repair(
        quantize_state_impl & qs,
        llama_model_loader & ml,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        const std::unordered_map<std::string, std::vector<float>> * imatrix_data,
        int nthread) {
    const llama_model_quantize_params * params = qs.params;
    const bool repair_enabled = params->quant_repair && (params->quant_repair_clipping || params->quant_repair_scale);
    if (!repair_enabled || params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_NONE) {
        return;
    }
    ensure_logit_gate_loaded(qs);
    const bool budget_limited = quant_budget_limited(params);
    const bool layer_output_proxy = params->quant_repair_scale;
    const repair_gate_settings gate_settings = repair_gate_for_context(qs, budget_limited, true);
    const float effective_min_error = gate_settings.min_error;
    const float effective_min_improvement = gate_settings.min_improvement;

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    int evaluated = 0;
    int clipped = 0;
    int skipped_low_error = 0;
    for (size_t i = 0; i < metadata.size(); ++i) {
        tensor_metadata & tm = metadata[i];
        ggml_tensor * tensor = tensors[i]->tensor;
        const int64_t ncols = tensor->ne[0];
        const int64_t nrows = ggml_nrows(tensor);
        if (!tm.allows_quantization ||
                tm.category == tensor_category::TOKEN_EMBD ||
                tm.category == tensor_category::OUTPUT ||
                !ggml_is_quantized(tm.target_type) ||
                ncols <= 0 || nrows <= 0) {
            continue;
        }

        const bool aggressive_candidate =
            budget_limited ||
            ggml_type_is_below_q4_for_policy(tm.target_type) ||
            tm.rd_distortion >= effective_min_error;
        if (!aggressive_candidate) {
            continue;
        }
        const float teacher_risk = logit_local_teacher_risk(qs, tm);
        const float local_min_error = effective_min_error / std::max(0.80f, teacher_risk);
        const float local_min_improvement = effective_min_improvement / std::sqrt(std::max(0.80f, teacher_risk));

        const float * imatrix = nullptr;
        if (imatrix_data) {
            auto it = imatrix_data->find(tm.remapped_imatrix_name);
            if (it != imatrix_data->end() && it->second.size() == (size_t) ncols * tensor->ne[2]) {
                imatrix = it->second.data();
            }
        }

        load_tensor_as_f32(ml, tensor, read_data, f32_conv, data, workers, nthread);

        const int sample_rows = (int) std::min<int64_t>(nrows, std::max(1, params->rd_sample_rows));
        const spqr_teacher_repair_result repair = evaluate_teacher_repair_clipping(
                tensor, data, imatrix, tm.target_type, sample_rows,
                local_min_error, local_min_improvement,
                params->quant_repair_clipping, params->quant_repair_scale);

        if (repair.base_error < 0.0f) {
            continue;
        }

        ++evaluated;
        tm.teacher_repair_error_before = repair.base_error;
        tm.teacher_repair_error_after = repair.repaired_error;
        if (repair.base_error < local_min_error) {
            ++skipped_low_error;
        }
        if (repair.clip_abs > 0.0f) {
            tm.teacher_repair_clip_abs = repair.clip_abs;
            tm.teacher_repair_source_gain = repair.source_gain;
            ++clipped;
            LLAMA_LOG_INFO("%s: quant-repair tensor=%-36s type=%-7s method=clipping clip_abs=%10.6g scale=%7.4f proxy_error=%10.6g -> %10.6g improvement=%6.3f teacher_risk=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(tm.target_type), repair.clip_abs, repair.source_gain,
                    repair.base_error, repair.repaired_error, repair.improvement, teacher_risk);
        } else if (repair.gain_repaired) {
            tm.teacher_repair_source_gain = repair.source_gain;
            ++clipped;
            LLAMA_LOG_INFO("%s: quant-repair tensor=%-36s type=%-7s method=scale scale=%7.4f proxy_error=%10.6g -> %10.6g improvement=%6.3f teacher_risk=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(tm.target_type), repair.source_gain,
                    repair.base_error, repair.repaired_error, repair.improvement, teacher_risk);
        } else if (params->print_rd_report) {
            LLAMA_LOG_INFO("%s: quant-repair tensor=%-36s type=%-7s selected=noop proxy_error=%10.6g best_error=%10.6g teacher_risk=%6.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(tm.target_type),
                    repair.base_error, repair.repaired_error, teacher_risk);
        }
    }

    if (evaluated > 0) {
        LLAMA_LOG_INFO("%s: quant-repair teacher evaluated=%d repaired=%d skipped_low_error=%d methods=%s%s layer_output_proxy=%s budget_limited=%s min_error=%g configured_min_error=%g min_improvement=%.3f configured_min_improvement=%.3f logit_gate=%s\n",
                __func__, evaluated, clipped, skipped_low_error,
                params->quant_repair_clipping ? "clipping" : "",
                params->quant_repair_scale ? ",scale" : "",
                layer_output_proxy ? "yes" : "no",
                budget_limited ? "yes" : "no",
                effective_min_error, params->quant_repair_min_error,
                effective_min_improvement, params->quant_repair_min_improvement,
                (params->logit_gate && qs.logit_report_available) ? (qs.logit_gate_pass ? "pass" : "fail") : "off");
    }
}

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
    const bool local_refinement_enabled =
        params->rd_local_refine_top_k > 0 && params->rd_local_refine_rows > max_sample_rows;
    if (params->rd_local_refine_top_k > 0 && !local_refinement_enabled) {
        LLAMA_LOG_WARN("%s: local RD refinement disabled because refine rows (%d) must exceed coarse sample rows (%d)\n",
                __func__, params->rd_local_refine_rows, max_sample_rows);
    }

    std::vector<no_init<uint8_t>> read_data;
    std::vector<no_init<float>> f32_conv;
    std::vector<float> data;
    std::vector<std::thread> workers;
    workers.reserve(nthread);

    std::unordered_map<std::string, const llama_model_quantize_rd_data *> precomputed;
    if (params->rd_profile) {
        for (const llama_model_quantize_rd_data * p = params->rd_profile; p->name != nullptr; ++p) {
            precomputed.emplace(p->name, p);
        }
    }

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

        const float distortion_weight = rd_distortion_weight(tm);

        float best_cost = std::numeric_limits<float>::infinity();
        ggml_type best_type = GGML_TYPE_COUNT;
        float best_distortion = 0.0f;
        float best_bpw = 0.0f;
        bool used_precomputed = false;

        auto profile_it = precomputed.find(tm.name);
        if (profile_it != precomputed.end() &&
                profile_it->second->ncols == ncols &&
                profile_it->second->nrows == nrows) {
            const llama_model_quantize_rd_data * profile = profile_it->second;
            bool profile_has_default = false;
            for (size_t candidate_i = 0; candidate_i < profile->size; ++candidate_i) {
                const ggml_type candidate = profile->types[candidate_i];
                const float distortion = profile->distortions[candidate_i];
                profile_has_default |= candidate == default_type;
                if (candidate >= GGML_TYPE_COUNT || distortion < 0.0f ||
                        !ggml_is_quantized(candidate) || ncols % ggml_blck_size(candidate) != 0) {
                    continue;
                }
                const float bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
                const float cost = distortion_weight * distortion + params->rd_lambda * bpw;
                tm.rd_candidates.push_back({
                    candidate,
                    distortion_weight * distortion,
                    distortion,
                    bpw,
                    (size_t) nrows * ggml_row_size(candidate, ncols),
                });
                if (params->print_rd_report) {
                    LLAMA_LOG_INFO("%s: rd-profile   tensor=%-36s type=%-7s distortion=%10.6g weight=%5.2f activity_risk=%5.2f bpw=%6.3f cost=%10.6g\n",
                            __func__, tm.name.c_str(), ggml_type_name(candidate), distortion,
                            distortion_weight, tm.activity_risk, bpw, cost);
                }
                if (cost < best_cost) {
                    best_cost = cost;
                    best_type = candidate;
                    best_distortion = distortion;
                    best_bpw = bpw;
                }
            }
            used_precomputed = profile_has_default && best_type != GGML_TYPE_COUNT;
            if (!used_precomputed) {
                best_cost = std::numeric_limits<float>::infinity();
                best_type = GGML_TYPE_COUNT;
                tm.rd_candidates.clear();
            }
        }

        const bool supplement_iq3_from_sampling = params->rd_include_iq3 && used_precomputed &&
                std::none_of(tm.rd_candidates.begin(), tm.rd_candidates.end(), [] (const rd_candidate & candidate) {
                    return candidate.type == GGML_TYPE_IQ3_S;
                });

        if (!used_precomputed || supplement_iq3_from_sampling) {
            load_tensor_as_f32(ml, tensor, read_data, f32_conv, data, workers, nthread);
        }

        const int sample_rows = (int) std::min<int64_t>(nrows, max_sample_rows);
        std::map<ggml_type, rd_block_stats> sampled_distortions;
        auto evaluate_candidate = [&] (ggml_type candidate) {
            if (sampled_distortions.count(candidate)) {
                return;
            }
            const rd_block_stats stats = sample_rd_candidate(tensor, data, imatrix, candidate, sample_rows);
            sampled_distortions[candidate] = stats;
            if (stats.aggregate < 0.0f) {
                return;
            }
            const float distortion = stats.aggregate;
            const float bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
            const float cost = distortion_weight * distortion + params->rd_lambda * bpw;
            tm.rd_candidates.push_back({
                candidate,
                distortion_weight * distortion,
                distortion,
                bpw,
                (size_t) nrows * ggml_row_size(candidate, ncols),
            });

            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: rd-candidate tensor=%-36s type=%-7s blocks=%3d aggregate_distortion=%10.6g weight=%5.2f activity_risk=%5.2f bpw=%6.3f cost=%10.6g\n",
                        __func__, tm.name.c_str(), ggml_type_name(candidate), sample_rows,
                        distortion, distortion_weight, tm.activity_risk, bpw, cost);
            }

            if (cost < best_cost) {
                best_cost = cost;
                best_type = candidate;
                best_distortion = distortion;
                best_bpw = bpw;
            }
        };

        if (supplement_iq3_from_sampling) {
            evaluate_candidate(GGML_TYPE_IQ3_S);
        }

        if (!used_precomputed) {
            const std::vector<ggml_type> primary_ladder = rd_primary_k_ladder(params, default_type);
            if (params->rd_include_iq3) {
                evaluate_candidate(GGML_TYPE_IQ3_S);
            }
            evaluate_candidate(GGML_TYPE_Q3_K);
            evaluate_candidate(GGML_TYPE_Q5_K);
            evaluate_candidate(default_type);
            const float q3 = sampled_distortions.count(GGML_TYPE_Q3_K) ? sampled_distortions[GGML_TYPE_Q3_K].aggregate : -1.0f;
            const float q5 = sampled_distortions.count(GGML_TYPE_Q5_K) ? sampled_distortions[GGML_TYPE_Q5_K].aggregate : -1.0f;
            // Conservative POC thresholds: fill the middle of a meaningful Q3-to-Q5
            // quality jump, and extend upward when Q5 still leaves visible distortion.
            if (q3 < 0.0f || q5 < 0.0f || q3 > 0.01f || q5 < 0.80f * q3) {
                evaluate_candidate(GGML_TYPE_Q4_K);
            }
            if (q5 < 0.0f || q5 > 0.003f) {
                evaluate_candidate(GGML_TYPE_Q6_K);
            }
            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: rd-ladder    tensor=%-36s ladder=%s\n",
                        __func__, tm.name.c_str(), format_type_list(primary_ladder).c_str());
            }
            const rd_block_stats q3_stats = sampled_distortions.count(GGML_TYPE_Q3_K) ?
                sampled_distortions[GGML_TYPE_Q3_K] : rd_block_stats {};
            const float tail_ratio = q3_stats.mean > 0.0f ?
                std::max(0.0f, q3_stats.worst / q3_stats.mean - 1.0f) : 0.0f;
            const float curve_gain = q3 > 0.0f && q5 >= 0.0f ?
                std::max(0.0f, 1.0f - q5 / q3) : 0.0f;
            tm.rd_refinement_score =
                0.45f * tail_ratio +
                0.35f * curve_gain +
                0.20f * std::max(0.0f, tm.activity_risk - 1.0f);
        }

        if (best_type != GGML_TYPE_COUNT) {
            tm.rd_type = best_type;
            tm.rd_cost = best_cost;
            tm.rd_distortion = best_distortion;
            tm.rd_bpw = best_bpw;
            tm.rd_from_profile = used_precomputed;
            ++qs.n_rd_selected;

            if (params->print_rd_report) {
                LLAMA_LOG_INFO("%s: rd-selected  tensor=%-36s type=%-7s source=%-9s candidates=%2d distortion=%10.6g bpw=%6.3f cost=%10.6g\n",
                        __func__, tm.name.c_str(), ggml_type_name(best_type),
                        used_precomputed ? "profile" : "sampled", (int) tm.rd_candidates.size(),
                        best_distortion, best_bpw, best_cost);
            }
        }
    }

    int refined_count = 0;
    if (local_refinement_enabled) {
        std::vector<size_t> refinement_candidates;
        for (size_t i = 0; i < metadata.size(); ++i) {
            const tensor_metadata & tm = metadata[i];
            if (!tm.rd_from_profile && !tm.rd_candidates.empty() && tm.rd_type != GGML_TYPE_COUNT) {
                refinement_candidates.push_back(i);
            }
        }
        std::sort(refinement_candidates.begin(), refinement_candidates.end(), [&] (size_t a, size_t b) {
            if (metadata[a].rd_refinement_score != metadata[b].rd_refinement_score) {
                return metadata[a].rd_refinement_score > metadata[b].rd_refinement_score;
            }
            return metadata[a].name < metadata[b].name;
        });
        refinement_candidates.resize(std::min(refinement_candidates.size(), (size_t) params->rd_local_refine_top_k));

        for (size_t rank = 0; rank < refinement_candidates.size(); ++rank) {
            const size_t i = refinement_candidates[rank];
            tensor_metadata & tm = metadata[i];
            ggml_tensor * tensor = tensors[i]->tensor;
            const int64_t ncols = tensor->ne[0];
            const int64_t nrows = ggml_nrows(tensor);
            const int sample_rows = (int) std::min<int64_t>(nrows, params->rd_local_refine_rows);

            const float * imatrix = nullptr;
            if (imatrix_data) {
                auto it = imatrix_data->find(tm.remapped_imatrix_name);
                if (it != imatrix_data->end() && it->second.size() == (size_t) ncols * tensor->ne[2]) {
                    imatrix = it->second.data();
                }
            }

            const float distortion_weight = rd_distortion_weight(tm);

            load_tensor_as_f32(ml, tensor, read_data, f32_conv, data, workers, nthread);

            std::vector<rd_candidate> refined_candidates;
            float best_cost = std::numeric_limits<float>::infinity();
            ggml_type best_type = GGML_TYPE_COUNT;
            float best_distortion = 0.0f;
            float best_bpw = 0.0f;
            std::vector<ggml_type> candidate_types = rd_primary_k_ladder(params, default_type);
            for (ggml_type candidate : candidate_types) {
                const rd_block_stats stats = sample_rd_candidate(tensor, data, imatrix, candidate, sample_rows);
                if (stats.aggregate < 0.0f) {
                    continue;
                }
                const float bpw = (float) ggml_row_size(candidate, ncols) * 8.0f / ncols;
                const float cost = distortion_weight * stats.aggregate + params->rd_lambda * bpw;
                refined_candidates.push_back({
                    candidate,
                    distortion_weight * stats.aggregate,
                    stats.aggregate,
                    bpw,
                    (size_t) nrows * ggml_row_size(candidate, ncols),
                });
                if (cost < best_cost) {
                    best_cost = cost;
                    best_type = candidate;
                    best_distortion = stats.aggregate;
                    best_bpw = bpw;
                }
                if (params->print_rd_refinement_report) {
                    LLAMA_LOG_INFO("%s: rd-refine-candidate rank=%3d tensor=%-36s type=%-7s rows=%3d distortion=%10.6g bpw=%6.3f cost=%10.6g\n",
                            __func__, (int) rank + 1, tm.name.c_str(), ggml_type_name(candidate),
                            sample_rows, stats.aggregate, bpw, cost);
                }
            }

            if (best_type == GGML_TYPE_COUNT) {
                continue;
            }

            const ggml_type coarse_type = tm.rd_type;
            tm.rd_candidates = std::move(refined_candidates);
            tm.rd_type = best_type;
            tm.rd_cost = best_cost;
            tm.rd_distortion = best_distortion;
            tm.rd_bpw = best_bpw;
            ++refined_count;

            if (params->print_rd_refinement_report) {
                LLAMA_LOG_INFO("%s: rd-refined   rank=%3d tensor=%-36s score=%8.4f rows=%3d coarse=%-7s selected=%-7s candidates=%2d\n",
                        __func__, (int) rank + 1, tm.name.c_str(), tm.rd_refinement_score, sample_rows,
                        ggml_type_name(coarse_type), ggml_type_name(best_type), (int) tm.rd_candidates.size());
            }
        }
    }

    LLAMA_LOG_INFO("%s: rate-distortion analysis selected existing quant types for %d tensor(s) (lambda=%.6g, max_sample_rows=%d, profile_entries=%d, locally_refined=%d)\n",
            __func__, qs.n_rd_selected, params->rd_lambda, max_sample_rows, (int) precomputed.size(), refined_count);
}

static const rd_candidate * select_rd_candidate(const tensor_metadata & tm, float lambda, float budget_bias = 1.0f) {
    const rd_candidate * best = nullptr;
    float best_cost = std::numeric_limits<float>::infinity();
    for (const rd_candidate & candidate : tm.rd_candidates) {
        const float cost = candidate.weighted_distortion + lambda * budget_bias * candidate.bpw;
        if (cost < best_cost || (cost == best_cost && best && candidate.bytes < best->bytes)) {
            best = &candidate;
            best_cost = cost;
        }
    }
    return best;
}

static void apply_rd_soft_target(
        quantize_state_impl & qs,
        const std::vector<const llama_model_loader::llama_tensor_weight *> & tensors,
        std::vector<tensor_metadata> & metadata,
        int64_t n_model_elements) {
    const llama_model_quantize_params * params = qs.params;
    if (params->rd_target_bpw <= 0.0f && params->rd_target_size_mib <= 0.0f) {
        return;
    }

    const size_t target_bytes = params->rd_target_bpw > 0.0f ?
        (size_t) std::ceil(params->rd_target_bpw * n_model_elements / 8.0) :
        (size_t) std::ceil(params->rd_target_size_mib * 1024.0 * 1024.0);
    size_t fixed_bytes = 0;
    std::vector<size_t> allocatable;
    const int max_layer = max_transformer_layer(metadata);

    for (size_t i = 0; i < metadata.size(); ++i) {
        const ggml_tensor * tensor = tensors[i]->tensor;
        const tensor_metadata & tm = metadata[i];
        // A differing target means a manual override, safety policy, or compatibility fallback won.
        if (!tm.rd_candidates.empty() && tm.rd_type == tm.target_type) {
            allocatable.push_back(i);
        } else {
            fixed_bytes += tm.target_type == tensor->type ?
                ggml_nbytes(tensor) :
                (size_t) ggml_nrows(tensor) * ggml_row_size(tm.target_type, tensor->ne[0]);
        }
    }

    auto estimate = [&] (float lambda, bool apply, float * distortion_out = nullptr) {
        size_t total = fixed_bytes;
        float total_distortion = 0.0f;
        for (size_t i : allocatable) {
            tensor_metadata & tm = metadata[i];
            const float budget_bias = lambda > 0.0f ? budget_layer_position_bias(tm, max_layer) : 1.0f;
            const rd_candidate * selected = select_rd_candidate(tm, lambda, budget_bias);
            if (!selected) {
                continue;
            }
            total += selected->bytes;
            total_distortion += selected->weighted_distortion;
            if (apply) {
                tm.rd_type = selected->type;
                tm.target_type = selected->type;
                tm.rd_distortion = selected->distortion;
                tm.rd_bpw = selected->bpw;
                tm.rd_cost = selected->weighted_distortion + lambda * budget_bias * selected->bpw;
            }
        }
        if (distortion_out) {
            *distortion_out = total_distortion;
        }
        return total;
    };

    float base_distortion = 0.0f;
    const size_t highest_quality_size = estimate(0.0f, false, &base_distortion);
    float high = std::max(params->rd_lambda, 1.0e-6f);
    size_t pressure_size = estimate(high, false);
    while (pressure_size > target_bytes && high < 1.0e6f) {
        high *= 2.0f;
        pressure_size = estimate(high, false);
    }
    float selected_lambda = 0.0f;
    bool budget_limited = false;

    if (highest_quality_size > target_bytes) {
        if (pressure_size <= target_bytes) {
            float low = 0.0f;
            for (int iteration = 0; iteration < 48; ++iteration) {
                const float mid = low + (high - low) * 0.5f;
                if (estimate(mid, false) <= target_bytes) {
                    high = mid;
                } else {
                    low = mid;
                }
            }
            selected_lambda = high;
        } else {
            selected_lambda = high;
            budget_limited = true;
        }
    }

    float selected_distortion = 0.0f;
    const size_t achieved_bytes = estimate(selected_lambda, true, &selected_distortion);
    qs.rd_allocation_lambda = selected_lambda;
    qs.rd_budget_distortion_base = base_distortion;
    qs.rd_budget_distortion_selected = selected_distortion;
    qs.rd_target_bytes = target_bytes;
    qs.rd_estimated_bytes = achieved_bytes;
    qs.rd_quality_limited = budget_limited;

    const float distortion_delta = selected_distortion - base_distortion;
    const double saved_mib = highest_quality_size > achieved_bytes ?
        (double) (highest_quality_size - achieved_bytes)/1024.0/1024.0 : 0.0;
    LLAMA_LOG_INFO("%s: bounded RD budget target=%8.2f MiB estimated=%8.2f MiB difference=%+.2f MiB lambda=%.6g status=%s allocatable=%d quality_cost=%10.6g saved_vs_hq=%8.2f MiB cost_per_mib=%10.6g bottom_first_bias=%s bias_range=0.94..1.08\n",
            __func__,
            target_bytes/1024.0/1024.0,
            achieved_bytes/1024.0/1024.0,
            ((double) achieved_bytes - target_bytes)/1024.0/1024.0,
            selected_lambda,
            budget_limited ? "bounded-limit" : "target-met",
            (int) allocatable.size(),
            distortion_delta,
            saved_mib,
            saved_mib > 0.0 ? distortion_delta / saved_mib : 0.0f,
            selected_lambda > 0.0f ? "on" : "off");

    if (params->print_rd_allocation_report) {
        for (size_t i : allocatable) {
            const tensor_metadata & tm = metadata[i];
            const float budget_bias = selected_lambda > 0.0f ? budget_layer_position_bias(tm, max_layer) : 1.0f;
            LLAMA_LOG_INFO("%s: rd-allocation tensor=%-36s selected=%-7s distortion=%10.6g bpw=%6.3f bytes=%zu budget_bias=%5.3f\n",
                    __func__, tm.name.c_str(), ggml_type_name(tm.rd_type), tm.rd_distortion, tm.rd_bpw,
                    (size_t) ggml_nrows(tensors[i]->tensor) * ggml_row_size(tm.rd_type, tensors[i]->tensor->ne[0]),
                    budget_bias);
        }
    }
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
    quant_log_section("Quant input");
    LLAMA_LOG_INFO("%s: output_ftype=%d dry_run=%s mixed_policy=%d rd_guided=%s repair=%s logit_search=%s\n",
            __func__,
            ftype,
            params->dry_run ? "yes" : "no",
            (int) params->mixed_quant_policy,
            params->rd_guided ? "yes" : "no",
            params->quant_repair ? "yes" : "no",
            params->logit_guided_search ? "yes" : "no");
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
        metadata[i].requested_type = GGML_TYPE_COUNT;
        metadata[i].compatibility_fallback_type = GGML_TYPE_COUNT;
        metadata[i].had_shape_fallback = false;
        metadata[i].fallback_ncols = 0;
        metadata[i].fallback_required_block_size = 0;
        metadata[i].teacher_repair_clip_abs = 0.0f;
        metadata[i].teacher_repair_source_gain = 1.0f;
        metadata[i].teacher_repair_error_before = 0.0f;
        metadata[i].teacher_repair_error_after = 0.0f;
    }

    int source_f32 = 0;
    int source_f16 = 0;
    int source_bf16 = 0;
    int source_quantized_convertible = 0;
    int source_quantized_unsupported = 0;
    int source_other = 0;
    std::map<ggml_type, int> auto_requantize_types;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const ggml_tensor * tensor = tensors[i]->tensor;
        switch (tensor->type) {
            case GGML_TYPE_F32:  ++source_f32;  break;
            case GGML_TYPE_F16:  ++source_f16;  break;
            case GGML_TYPE_BF16: ++source_bf16; break;
            default:
                if (ggml_is_quantized(tensor->type)) {
                    if (tensor_type_can_convert_to_f32(tensor->type)) {
                        ++source_quantized_convertible;
                        if (metadata[i].allows_quantization) {
                            ++auto_requantize_types[tensor->type];
                        }
                    } else {
                        ++source_quantized_unsupported;
                        if (metadata[i].allows_quantization) {
                            throw std::runtime_error(format(
                                    "cannot automatically requantize tensor %s from source type %s: source type has no to_float converter",
                                    tensor->name, ggml_type_name(tensor->type)));
                        }
                    }
                } else {
                    ++source_other;
                    if (metadata[i].allows_quantization && !tensor_type_can_convert_to_f32(tensor->type)) {
                        throw std::runtime_error(format(
                                "cannot automatically requantize tensor %s from non-floating source type %s",
                                tensor->name, ggml_type_name(tensor->type)));
                    }
                }
                break;
        }
    }
    if (source_f32 > 0 || source_bf16 > 0 || source_quantized_convertible > 0 ||
            source_quantized_unsupported > 0 || source_other > 0) {
        LLAMA_LOG_INFO("%s: source tensor types: f32=%d f16=%d bf16=%d quantized_convertible=%d quantized_unsupported=%d other=%d\n",
                __func__, source_f32, source_f16, source_bf16,
                source_quantized_convertible, source_quantized_unsupported, source_other);
    }
    for (const auto & kv : auto_requantize_types) {
        LLAMA_LOG_WARN("%s: automatically allowing requantize from source type %s for %d quantizable tensor(s) because to_float is available\n",
                __func__, ggml_type_name(kv.first), kv.second);
    }

    quant_log_section("Analysis and allocation");
    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA ||
            params->print_layer_delta_report ||
            params->rd_guided) {
        init_spqr_guided_sensitivity(qs, metadata, imatrix_data);
        if (params->spqr_block_report || params->spqr_block_scoring) {
            init_spqr_guided_block_scoring(qs, metadata, imatrix_data, params->spqr_block_size, params->spqr_block_scoring);
        }
    }
    if (params->rd_guided ||
            spqr_layer_delta_guidance_enabled(params) ||
            params->print_layer_delta_report) {
        init_activity_profile(metadata, params);
    }
    if (spqr_layer_delta_guidance_enabled(params) || params->print_layer_delta_report) {
        const int loaded = init_precomputed_layer_delta_analysis(qs, metadata, params->print_layer_delta_report);
        if (loaded == 0) {
            init_layer_delta_analysis(qs, ml, tensors, metadata, nthread, params->print_layer_delta_report);
        }
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

    quant_log_section("Initial tensor type selection");
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
    }

    apply_shape_aware_scalar_fallback(qs, ml, tensors, metadata, imatrix_data, nthread);

    for (size_t i = 0; i < tensors.size(); ++i) {
        const struct ggml_tensor * tensor = tensors[i]->tensor;
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

    quant_log_section("Repair and budget passes");
    apply_rd_soft_target(qs, tensors, metadata, ml.n_elements);
    apply_budget_first_type_cap(qs, ml, tensors, metadata, imatrix_data, nthread);
    apply_spqr_repair(qs, ml, tensors, metadata, imatrix_data, nthread);
    apply_budget_repair_shrink(qs, ml, tensors, metadata, imatrix_data, nthread);
    apply_logit_guided_budget_buyback(qs, ml, tensors, metadata, imatrix_data, nthread);
    apply_quality_precision_validation(qs, ml, tensors, metadata, imatrix_data, nthread);
    apply_spqr_teacher_repair(qs, ml, tensors, metadata, imatrix_data, nthread);
    print_compression_opportunity_report(qs, ml, tensors, metadata, imatrix_data, nthread);
    if (params->rd_target_bpw > 0.0f || params->rd_target_size_mib > 0.0f ||
            params->quant_repair) {
        for (size_t i = 0; i < tensors.size(); ++i) {
            metadata[i].requires_imatrix = tensor_requires_imatrix(
                    tensors[i]->tensor->name, metadata[i].target_type, ftype);
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
    std::vector<no_init<float>> f32_repair_buf;

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

    quant_log_section(params->dry_run ? "Dry-run tensor accounting" : "Tensor quantization");
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
                ml.load_data_for(tensor);
            } else if (tensor->data == nullptr) {
                ml.load_data_for(tensor);
            }
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
                        spqr_layer_delta_guidance_enabled(params)) && tm.allows_quantization) {
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
                } else if (!tensor_type_can_convert_to_f32(tensor->type)) {
                    throw std::runtime_error(format(
                            "cannot requantize tensor %s from source type %s: source type has no to_float converter",
                            tensor->name, ggml_type_name(tensor->type)));
                } else {
                    if (ggml_is_quantized(tensor->type) && !params->allow_requantize) {
                        LLAMA_LOG_WARN("auto-requantize source=%s .. ", ggml_type_name(tensor->type));
                    }
                    llama_tensor_dequantize_impl(tensor, f32_conv_buf, workers, nelements, nthread);
                    f32_data = (float *) f32_conv_buf.data();
                }

                if (tm.teacher_repair_clip_abs > 0.0f || tm.teacher_repair_source_gain != 1.0f) {
                    f32_repair_buf.resize((size_t) nelements);
                    float * repair_data = (float *) f32_repair_buf.data();
                    for (int64_t k = 0; k < nelements; ++k) {
                        const float clipped = tm.teacher_repair_clip_abs > 0.0f ?
                                std::max(-tm.teacher_repair_clip_abs, std::min(tm.teacher_repair_clip_abs, f32_data[k])) :
                                f32_data[k];
                        repair_data[k] = tm.teacher_repair_source_gain * clipped;
                    }
                    f32_data = repair_data;
                    LLAMA_LOG_INFO("teacher-repair clip_abs=%g gain=%g .. ",
                            tm.teacher_repair_clip_abs, tm.teacher_repair_source_gain);
                    fflush(stdout);
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
                        spqr_layer_delta_guidance_enabled(params)) && tm.allows_quantization) {
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

    quant_log_section("Quant summary");
    LLAMA_LOG_INFO("%s: model size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_org/1024.0/1024.0, total_size_org*8.0/ml.n_elements);
    LLAMA_LOG_INFO("%s: quant size  = %8.2f MiB (%.2f BPW)\n", __func__, total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);

    if (params->mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            spqr_layer_delta_guidance_enabled(params)) {
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
    if (spqr_layer_delta_guidance_enabled(params)) {
        LLAMA_LOG_INFO("%s: layer-delta-guidance summary: high_similarity=%d medium_similarity=%d low_similarity=%d anchors=%d anchor_tensors=%d demoted=%d\n",
                __func__, qs.n_delta_high, qs.n_delta_medium, qs.n_delta_low,
                qs.n_anchor_layers, qs.n_anchor_tensors, qs.n_delta_demoted);
    }
    if (params->rd_guided) {
        LLAMA_LOG_INFO("%s: sampled rate-distortion summary: selected=%d base_lambda=%.6g sample_rows=%d total_size=%8.2f MiB avg_bpw=%.2f\n",
                __func__, qs.n_rd_selected, params->rd_lambda, params->rd_sample_rows,
                total_size_new/1024.0/1024.0, total_size_new*8.0/ml.n_elements);
        if (qs.rd_target_bytes > 0) {
            const float quality_cost = qs.rd_budget_distortion_selected - qs.rd_budget_distortion_base;
            LLAMA_LOG_INFO("%s: bounded RD budget summary: target=%8.2f MiB actual=%8.2f MiB difference=%+.2f MiB selected_lambda=%.6g quality_cost=%10.6g status=%s bottom_first_bias=%s bias_range=0.94..1.08\n",
                    __func__,
                    qs.rd_target_bytes/1024.0/1024.0,
                    total_size_new/1024.0/1024.0,
                    ((double) total_size_new - qs.rd_target_bytes)/1024.0/1024.0,
                    qs.rd_allocation_lambda,
                    quality_cost,
                    qs.rd_quality_limited ? "bounded-limit" : "target-met",
                    qs.rd_allocation_lambda > 0.0f ? "on" : "off");
        }
    }
    if (params->logit_gate) {
        ensure_logit_gate_loaded(qs);
        if (qs.logit_report_available) {
            LLAMA_LOG_INFO("%s: logit-gate summary report=%s status=%s mode=%s damage=%10.6g mean_topk_kl=%10.6g argmax_flip_rate=%10.6g thresholds=(%g,%g,%g) tensor_deltas=%zu layer_deltas=%zu family_deltas=%zu local_allocator_prior=%s\n",
                    __func__, params->logit_report,
                    qs.logit_gate_pass ? "pass" : "fail",
                    qs.logit_report_paired ? "paired" : "candidate-only",
                    qs.logit_damage_score,
                    qs.logit_mean_topk_kl,
                    qs.logit_argmax_flip_rate,
                    params->logit_damage_threshold,
                    params->logit_kl_threshold,
                    params->logit_flip_threshold,
                    qs.logit_tensor_delta_mean_mse.size(),
                    qs.logit_layer_delta_mean_mse.size(),
                    qs.logit_family_delta_mean_mse.size(),
                    (qs.logit_report_paired && (!qs.logit_tensor_delta_mean_mse.empty() || !qs.logit_layer_delta_mean_mse.empty() || !qs.logit_family_delta_mean_mse.empty())) ? "on" : "off");
        } else {
            LLAMA_LOG_WARN("%s: logit-gate summary report=%s status=unavailable\n",
                    __func__, params->logit_report ? params->logit_report : "(null)");
        }
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
        /*.layer_delta_guidance        =*/ false,
        /*.spqr_block_scoring          =*/ false,
        /*.adaptive_anchors            =*/ false,
        /*.anchor_percentile           =*/ 90.0f,
        /*.print_anchor_report         =*/ false,
        /*.rd_guided                   =*/ false,
        /*.rd_include_iq3              =*/ false,
        /*.rd_lambda                   =*/ 0.002f,
        /*.rd_sample_rows              =*/ 8,
        /*.print_rd_report             =*/ false,
        /*.rd_target_bpw               =*/ 0.0f,
        /*.rd_target_size_mib          =*/ 0.0f,
        /*.print_rd_allocation_report  =*/ false,
        /*.rd_local_refine_top_k       =*/ 0,
        /*.rd_local_refine_rows        =*/ 32,
        /*.print_rd_refinement_report  =*/ false,
        /*.print_compression_opportunity_report =*/ false,
        /*.rd_profile                  =*/ nullptr,
        /*.layer_delta_profile         =*/ nullptr,
        /*.activity_profile            =*/ nullptr,
        /*.quant_repair                =*/ false,
        /*.quant_repair_clipping       =*/ true,
        /*.quant_repair_gain           =*/ true,
        /*.quant_repair_scale          =*/ true,
        /*.quant_teacher_aware        =*/ true,
        /*.quant_teacher_aware_mix    =*/ 0.35f,
        /*.quant_teacher_aware_block_mix =*/ 0.10f,
        /*.quant_teacher_aware_rank_mix =*/ 0.05f,
        /*.quant_teacher_aware_top_k  =*/ 8,
        /*.quant_repair_accept_ratio   =*/ 1.05f,
        /*.quant_repair_max_error      =*/ 0.001f,
        /*.quant_repair_min_error      =*/ 0.002f,
        /*.quant_repair_min_improvement =*/ 0.05f,
        /*.logit_report                =*/ nullptr,
        /*.logit_gate                  =*/ false,
        /*.logit_damage_threshold      =*/ 0.02f,
        /*.logit_kl_threshold          =*/ 0.01f,
        /*.logit_flip_threshold        =*/ 0.02f,
        /*.logit_guided_search         =*/ false,
        /*.logit_search_top_k          =*/ 24,
        /*.logit_search_promote_budget_mib =*/ 64.0f,
        /*.logit_search_package_max_items =*/ 3,
        /*.logit_search_package_pool_size =*/ 24,
        /*.logit_search_package_overshoot_weight =*/ 0.05f,
        /*.logit_search_package_same_layer_penalty =*/ 0.01f,
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
        metadata[i].requested_type = GGML_TYPE_COUNT;
        metadata[i].compatibility_fallback_type = GGML_TYPE_COUNT;
        metadata[i].had_shape_fallback = false;
        metadata[i].fallback_ncols = 0;
        metadata[i].fallback_required_block_size = 0;
        metadata[i].teacher_repair_clip_abs = 0.0f;
        metadata[i].teacher_repair_source_gain = 1.0f;
        metadata[i].teacher_repair_error_before = 0.0f;
        metadata[i].teacher_repair_error_after = 0.0f;
    }
    if (local_params.mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_GUIDED ||
            local_params.mixed_quant_policy == LLAMA_MIXED_QUANT_POLICY_SPQR_LAYER_DELTA) {
        init_spqr_guided_sensitivity(*qs, metadata, nullptr);
        if (local_params.spqr_block_report || local_params.spqr_block_scoring) {
            init_spqr_guided_block_scoring(*qs, metadata, nullptr, local_params.spqr_block_size, local_params.spqr_block_scoring);
        }
    }
    if (local_params.rd_guided ||
            spqr_layer_delta_guidance_enabled(&local_params) ||
            local_params.print_layer_delta_report) {
        init_activity_profile(metadata, &local_params);
    }

    ggml_type default_type = llama_ftype_get_default_type(ftype);

    // compute types
    for (size_t i = 0; i < n_tensors; i++) {
        result_types[i] = llama_tensor_get_type(*qs, &local_params, tensors[i], default_type, metadata[i]);
    }
}
