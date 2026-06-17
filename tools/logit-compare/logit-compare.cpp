#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct logit_compare_args {
    std::string teacher_model;
    std::string baseline_model;
    std::string student_model;
    std::string json_out;
    int top_k = 64;
    bool layer_attribution = false;
    int layer_sample_tokens = 8;
};

static void logit_log_section(const char * title) {
    LOG_INF("\n==== %s ====\n", title);
}

struct topk_entry {
    llama_token token;
    float logit;
};

struct topk_stats {
    std::vector<topk_entry> entries;
    llama_token argmax = LLAMA_TOKEN_NULL;
    float max_logit = -std::numeric_limits<float>::infinity();
    float second_logit = -std::numeric_limits<float>::infinity();
    double log_sum_exp = 0.0;
};

struct chunk_metrics {
    int chunk_index = 0;
    int compared_positions = 0;
    double mean_topk_kl = 0.0;
    double mean_topk_overlap = 0.0;
    double mean_rank_drift = 0.0;
    double mean_teacher_margin = 0.0;
    double weighted_topk_kl = 0.0;
    double mean_next_logprob_delta = 0.0;
    double mean_next_rank_drift = 0.0;
    double teacher_winner_in_student_topk = 0.0;
    double next_token_in_teacher_topk = 0.0;
    double next_token_in_student_topk = 0.0;
    double p50_topk_kl = 0.0;
    double p90_topk_kl = 0.0;
    double p95_topk_kl = 0.0;
    double max_topk_kl = 0.0;
    double p50_rank_drift = 0.0;
    double p90_rank_drift = 0.0;
    double p95_rank_drift = 0.0;
    double max_rank_drift = 0.0;
    int argmax_flips = 0;
};

struct global_metrics {
    int compared_positions = 0;
    int argmax_flips = 0;
    double sum_topk_kl = 0.0;
    double sum_topk_overlap = 0.0;
    double sum_rank_drift = 0.0;
    double sum_teacher_margin = 0.0;
    double sum_weighted_topk_kl = 0.0;
    double sum_margin_weight = 0.0;
    double sum_next_logprob_delta = 0.0;
    double sum_next_rank_drift = 0.0;
    double sum_teacher_winner_in_student_topk = 0.0;
    double sum_next_token_in_teacher_topk = 0.0;
    double sum_next_token_in_student_topk = 0.0;
    std::vector<double> topk_kl_values;
    std::vector<double> rank_drift_values;
    std::vector<double> teacher_margin_values;
    std::vector<double> next_logprob_delta_values;
    std::vector<chunk_metrics> chunks;
};

struct percentile_summary {
    double p50 = 0.0;
    double p90 = 0.0;
    double p95 = 0.0;
    double max = 0.0;
};

struct metric_summary {
    int compared_positions = 0;
    double mean_topk_kl = 0.0;
    double weighted_topk_kl = 0.0;
    double mean_topk_overlap = 0.0;
    double mean_rank_drift = 0.0;
    double teacher_margin_mean = 0.0;
    double mean_next_logprob_delta = 0.0;
    double mean_next_rank_drift = 0.0;
    double teacher_winner_in_student_topk = 0.0;
    double next_token_in_teacher_topk = 0.0;
    double next_token_in_student_topk = 0.0;
    double argmax_flip_rate = 0.0;
    percentile_summary topk_kl_percentiles;
    percentile_summary rank_drift_percentiles;
    percentile_summary teacher_margin_percentiles;
    percentile_summary next_logprob_delta_percentiles;
};

struct metric_delta {
    double delta_mean_topk_kl = 0.0;
    double delta_weighted_topk_kl = 0.0;
    double delta_mean_topk_overlap = 0.0;
    double delta_mean_rank_drift = 0.0;
    double delta_mean_next_logprob_delta = 0.0;
    double delta_mean_next_rank_drift = 0.0;
    double delta_teacher_winner_in_student_topk = 0.0;
    double delta_next_token_in_student_topk = 0.0;
    double delta_argmax_flip_rate = 0.0;
    double delta_p95_topk_kl = 0.0;
    double delta_p95_rank_drift = 0.0;
    double damage_score = 0.0;
};

struct attribution_sample {
    std::string name;
    int64_t cols = 0;
    int64_t rows = 0;
    int sampled_rows = 0;
    std::vector<float> values;
};

struct attribution_capture {
    bool enabled = false;
    int sample_tokens = 8;
    std::unordered_map<std::string, attribution_sample> samples;
    std::vector<float> scratch;
};

struct attribution_record {
    std::string key;
    int layer = -1;
    int compared_tensors = 0;
    int compared_values = 0;
    double sum_mse = 0.0;
    double sum_cosine_error = 0.0;
    double sum_norm_ratio_error = 0.0;
    double sum_abs_error = 0.0;
    double max_mse = 0.0;
};

struct attribution_group_summary {
    std::string key;
    int layer = -1;
    int compared_tensors = 0;
    int compared_values = 0;
    double mean_mse = 0.0;
    double mean_cosine_error = 0.0;
    double mean_norm_ratio_error = 0.0;
    double mean_abs_error = 0.0;
    double max_mse = 0.0;
};

struct attribution_delta_summary {
    std::string key;
    int layer = -1;
    double delta_mean_mse = 0.0;
    double delta_mean_cosine_error = 0.0;
    double delta_mean_norm_ratio_error = 0.0;
    double delta_mean_abs_error = 0.0;
    double delta_max_mse = 0.0;
};

struct attribution_report {
    std::unordered_map<std::string, attribution_record> tensors;
    std::unordered_map<std::string, attribution_record> families;
    std::unordered_map<int, attribution_record> layers;
};

struct compare_result {
    global_metrics metrics;
    attribution_report attribution;
};

struct per_position_metrics {
    double topk_kl = 0.0;
    double topk_overlap = 0.0;
    double rank_drift = 0.0;
    double teacher_margin = 0.0;
    double next_logprob_delta = 0.0;
    double next_rank_drift = 0.0;
    double margin_weight = 0.0;
    bool argmax_flip = false;
    bool teacher_winner_in_student_topk = false;
    bool next_token_in_teacher_topk = false;
    bool next_token_in_student_topk = false;
};

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s --teacher teacher-f16.gguf --student student-q4.gguf -f eval.txt -c 512 --logit-top-k 64\n", argv[0]);
    LOG("\n    %s --teacher teacher-f16.gguf --baseline baseline-q4.gguf --student candidate-q4.gguf -f eval.txt -c 512\n", argv[0]);
    LOG("\ncustom options:\n");
    LOG("    --teacher MODEL        teacher/reference model path\n");
    LOG("    --baseline MODEL       optional baseline student model for paired delta reporting\n");
    LOG("    --student MODEL        student/quantized model path\n");
    LOG("    --logit-top-k N        top-k set used for KL/overlap/rank metrics (default: 64)\n");
    LOG("    --layer-attribution    collect sampled hidden-state attribution by tensor/family/layer\n");
    LOG("    --layer-sample-tokens N sampled token rows per tensor when --layer-attribution is enabled (default: 8)\n");
    LOG("    --json-out FILE        optional JSON summary output path\n");
    LOG("\ncommon options:\n");
    LOG("    reuse standard evaluation flags such as -f, -c, -b, -t, -ngl, --chunks\n");
    LOG("\n");
}

static bool take_string_arg(int & i, int argc, char ** argv, std::string & out, const char * name) {
    if (i + 1 >= argc) {
        LOG_ERR("%s: missing value for %s\n", __func__, name);
        return false;
    }
    out = argv[++i];
    return true;
}

static bool take_int_arg(int & i, int argc, char ** argv, int & out, const char * name) {
    if (i + 1 >= argc) {
        LOG_ERR("%s: missing value for %s\n", __func__, name);
        return false;
    }
    try {
        out = std::stoi(argv[++i]);
    } catch (const std::exception &) {
        LOG_ERR("%s: invalid integer for %s: %s\n", __func__, name, argv[i]);
        return false;
    }
    return true;
}

static bool parse_logit_compare_args(int argc, char ** argv, logit_compare_args & lc_args, std::vector<std::string> & passthrough) {
    passthrough.clear();
    passthrough.emplace_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--teacher") {
            if (!take_string_arg(i, argc, argv, lc_args.teacher_model, "--teacher")) {
                return false;
            }
            continue;
        }
        if (arg == "--baseline") {
            if (!take_string_arg(i, argc, argv, lc_args.baseline_model, "--baseline")) {
                return false;
            }
            continue;
        }
        if (arg == "--student") {
            if (!take_string_arg(i, argc, argv, lc_args.student_model, "--student")) {
                return false;
            }
            continue;
        }
        if (arg == "--json-out") {
            if (!take_string_arg(i, argc, argv, lc_args.json_out, "--json-out")) {
                return false;
            }
            continue;
        }
        if (arg == "--logit-top-k") {
            if (!take_int_arg(i, argc, argv, lc_args.top_k, "--logit-top-k")) {
                return false;
            }
            continue;
        }
        if (arg == "--layer-attribution") {
            lc_args.layer_attribution = true;
            continue;
        }
        if (arg == "--layer-sample-tokens") {
            if (!take_int_arg(i, argc, argv, lc_args.layer_sample_tokens, "--layer-sample-tokens")) {
                return false;
            }
            continue;
        }
        if (arg == "-m" || arg == "--model") {
            LOG_ERR("%s: use --student instead of %s for this tool\n", __func__, arg.c_str());
            return false;
        }
        passthrough.push_back(arg);
    }

    if (lc_args.teacher_model.empty() || lc_args.student_model.empty()) {
        LOG_ERR("%s: both --teacher and --student are required\n", __func__);
        return false;
    }
    if (lc_args.top_k <= 0) {
        LOG_ERR("%s: --logit-top-k must be positive\n", __func__);
        return false;
    }
    if (lc_args.layer_sample_tokens <= 0) {
        LOG_ERR("%s: --layer-sample-tokens must be positive\n", __func__);
        return false;
    }

    passthrough.emplace_back("-m");
    passthrough.emplace_back(lc_args.student_model);
    return true;
}

static std::vector<char *> make_argv(std::vector<std::string> & args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (std::string & arg : args) {
        argv.push_back(arg.data());
    }
    return argv;
}

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

static bool starts_with(const std::string & s, const char * prefix) {
    const size_t n = std::strlen(prefix);
    return s.size() >= n && s.compare(0, n, prefix) == 0;
}

static bool parse_layered_tensor_name(const std::string & name, std::string & family, int & layer) {
    const size_t pos = name.rfind('-');
    if (pos == std::string::npos || pos + 1 >= name.size()) {
        family.clear();
        layer = -1;
        return false;
    }

    family = name.substr(0, pos);
    try {
        layer = std::stoi(name.substr(pos + 1));
    } catch (const std::exception &) {
        family.clear();
        layer = -1;
        return false;
    }
    return true;
}

static bool should_capture_attribution_tensor(const std::string & name) {
    if (name == "result_norm") {
        return true;
    }

    std::string family;
    int layer = -1;
    if (!parse_layered_tensor_name(name, family, layer)) {
        return false;
    }

    return family == "l_out" || family == "attn_out" || family == "ffn_out";
}

static std::vector<int64_t> select_sample_rows(int64_t n_rows, int sample_rows) {
    std::vector<int64_t> rows;
    if (n_rows <= 0 || sample_rows <= 0) {
        return rows;
    }

    const int take = std::min<int64_t>(n_rows, sample_rows);
    rows.reserve(take);

    if (take == 1) {
        rows.push_back(n_rows - 1);
        return rows;
    }

    for (int i = 0; i < take; ++i) {
        const double t = take > 1 ? (double) i / (take - 1) : 0.0;
        const int64_t row = std::min<int64_t>(n_rows - 1, (int64_t) std::llround(t * (n_rows - 1)));
        if (rows.empty() || rows.back() != row) {
            rows.push_back(row);
        }
    }

    for (int64_t row = n_rows - 1; (int) rows.size() < take && row >= 0; --row) {
        if (std::find(rows.begin(), rows.end(), row) == rows.end()) {
            rows.push_back(row);
        }
    }

    std::sort(rows.begin(), rows.end());
    return rows;
}

static bool capture_attribution_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * capture = static_cast<attribution_capture *>(user_data);
    if (!capture->enabled) {
        return false;
    }

    const std::string name = t->name ? t->name : "";
    if (ask) {
        return should_capture_attribution_tensor(name);
    }

    if (!should_capture_attribution_tensor(name) || t->type != GGML_TYPE_F32 || t->ne[0] <= 0 || t->ne[1] <= 0) {
        return true;
    }

    const std::vector<int64_t> rows = select_sample_rows(t->ne[1], capture->sample_tokens);
    if (rows.empty()) {
        return true;
    }

    const size_t n_bytes = ggml_nbytes(t);
    capture->scratch.resize(n_bytes);

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * src = nullptr;
    if (is_host) {
        src = static_cast<const uint8_t *>(t->data);
    } else {
        ggml_backend_tensor_get(t, capture->scratch.data(), 0, n_bytes);
        src = reinterpret_cast<const uint8_t *>(capture->scratch.data());
    }

    attribution_sample & sample = capture->samples[name];
    if (sample.values.empty()) {
        sample.name = name;
        sample.cols = t->ne[0];
        sample.rows = 0;
        sample.sampled_rows = 0;
    }
    if (sample.cols != t->ne[0]) {
        return true;
    }
    sample.rows += t->ne[1];
    sample.sampled_rows += (int) rows.size();
    sample.values.reserve(sample.values.size() + (size_t) sample.cols * rows.size());

    for (int64_t row : rows) {
        const size_t offset = (size_t) row * t->nb[1];
        const float * row_ptr = reinterpret_cast<const float *>(src + offset);
        sample.values.insert(sample.values.end(), row_ptr, row_ptr + sample.cols);
    }
    return true;
}

static void add_attribution_record(
        std::unordered_map<std::string, attribution_record> & table,
        const std::string & key,
        int layer,
        int compared_values,
        double mse,
        double cosine_error,
        double norm_ratio_error,
        double abs_error) {
    attribution_record & record = table[key];
    record.key = key;
    record.layer = layer;
    record.compared_tensors += 1;
    record.compared_values += compared_values;
    record.sum_mse += mse * compared_values;
    record.sum_cosine_error += cosine_error * compared_values;
    record.sum_norm_ratio_error += norm_ratio_error * compared_values;
    record.sum_abs_error += abs_error * compared_values;
    record.max_mse = std::max(record.max_mse, mse);
}

static void add_attribution_record(
        std::unordered_map<int, attribution_record> & table,
        int key,
        int compared_values,
        double mse,
        double cosine_error,
        double norm_ratio_error,
        double abs_error) {
    attribution_record & record = table[key];
    record.key = std::to_string(key);
    record.layer = key;
    record.compared_tensors += 1;
    record.compared_values += compared_values;
    record.sum_mse += mse * compared_values;
    record.sum_cosine_error += cosine_error * compared_values;
    record.sum_norm_ratio_error += norm_ratio_error * compared_values;
    record.sum_abs_error += abs_error * compared_values;
    record.max_mse = std::max(record.max_mse, mse);
}

static void accumulate_attribution(
        attribution_report & report,
        const attribution_capture & teacher,
        const attribution_capture & student) {
    for (const auto & kv : teacher.samples) {
        const std::string & name = kv.first;
        const attribution_sample & lhs = kv.second;
        const auto it = student.samples.find(name);
        if (it == student.samples.end()) {
            continue;
        }

        const attribution_sample & rhs = it->second;
        const size_t n = std::min(lhs.values.size(), rhs.values.size());
        if (n == 0) {
            continue;
        }

        double sum_diff2 = 0.0;
        double sum_abs = 0.0;
        double dot = 0.0;
        double lhs_norm2 = 0.0;
        double rhs_norm2 = 0.0;

        for (size_t i = 0; i < n; ++i) {
            const double a = lhs.values[i];
            const double b = rhs.values[i];
            const double diff = a - b;
            sum_diff2 += diff * diff;
            sum_abs += std::abs(diff);
            dot += a * b;
            lhs_norm2 += a * a;
            rhs_norm2 += b * b;
        }

        const double denom = std::max<size_t>(1, n);
        const double mse = sum_diff2 / denom;
        const double abs_error = sum_abs / denom;
        const double cosine = dot / std::max(1e-12, std::sqrt(lhs_norm2) * std::sqrt(rhs_norm2));
        const double cosine_error = 1.0 - cosine;
        const double norm_ratio_error = std::abs(std::sqrt(rhs_norm2 / std::max(lhs_norm2, 1e-12)) - 1.0);

        std::string family;
        int layer = -1;
        if (!parse_layered_tensor_name(name, family, layer)) {
            family = name;
        }

        add_attribution_record(report.tensors, name, layer, (int) n, mse, cosine_error, norm_ratio_error, abs_error);
        add_attribution_record(report.families, family, -1, (int) n, mse, cosine_error, norm_ratio_error, abs_error);
        if (layer >= 0) {
            add_attribution_record(report.layers, layer, (int) n, mse, cosine_error, norm_ratio_error, abs_error);
        }
    }
}

static std::vector<attribution_group_summary> summarize_attribution(
        const std::unordered_map<std::string, attribution_record> & table) {
    std::vector<attribution_group_summary> out;
    out.reserve(table.size());
    for (const auto & kv : table) {
        const attribution_record & record = kv.second;
        const double denom = std::max(1, record.compared_values);
        out.push_back({
            record.key,
            record.layer,
            record.compared_tensors,
            record.compared_values,
            record.sum_mse / denom,
            record.sum_cosine_error / denom,
            record.sum_norm_ratio_error / denom,
            record.sum_abs_error / denom,
            record.max_mse,
        });
    }

    std::sort(out.begin(), out.end(), [] (const attribution_group_summary & a, const attribution_group_summary & b) {
        if (a.mean_mse != b.mean_mse) {
            return a.mean_mse > b.mean_mse;
        }
        return a.key < b.key;
    });
    return out;
}

static std::vector<attribution_group_summary> summarize_attribution(
        const std::unordered_map<int, attribution_record> & table) {
    std::vector<attribution_group_summary> out;
    out.reserve(table.size());
    for (const auto & kv : table) {
        const attribution_record & record = kv.second;
        const double denom = std::max(1, record.compared_values);
        out.push_back({
            record.key,
            record.layer,
            record.compared_tensors,
            record.compared_values,
            record.sum_mse / denom,
            record.sum_cosine_error / denom,
            record.sum_norm_ratio_error / denom,
            record.sum_abs_error / denom,
            record.max_mse,
        });
    }

    std::sort(out.begin(), out.end(), [] (const attribution_group_summary & a, const attribution_group_summary & b) {
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.key < b.key;
    });
    return out;
}

static std::vector<attribution_delta_summary> compute_attribution_delta(
        const std::vector<attribution_group_summary> & baseline,
        const std::vector<attribution_group_summary> & candidate) {
    std::unordered_map<std::string, attribution_group_summary> baseline_by_key;
    baseline_by_key.reserve(baseline.size());
    for (const auto & item : baseline) {
        baseline_by_key[item.key] = item;
    }

    std::vector<attribution_delta_summary> out;
    out.reserve(candidate.size());
    for (const auto & item : candidate) {
        const auto it = baseline_by_key.find(item.key);
        if (it == baseline_by_key.end()) {
            continue;
        }

        const attribution_group_summary & base = it->second;
        out.push_back({
            item.key,
            item.layer,
            item.mean_mse - base.mean_mse,
            item.mean_cosine_error - base.mean_cosine_error,
            item.mean_norm_ratio_error - base.mean_norm_ratio_error,
            item.mean_abs_error - base.mean_abs_error,
            item.max_mse - base.max_mse,
        });
    }

    std::sort(out.begin(), out.end(), [] (const attribution_delta_summary & a, const attribution_delta_summary & b) {
        if (a.delta_mean_mse != b.delta_mean_mse) {
            return a.delta_mean_mse > b.delta_mean_mse;
        }
        return a.key < b.key;
    });
    return out;
}

static percentile_summary summarize_percentiles(const std::vector<double> & values) {
    percentile_summary summary;
    if (values.empty()) {
        return summary;
    }

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    auto percentile = [&sorted](double q) {
        const double idx = q * (sorted.size() - 1);
        const size_t lo = (size_t) std::floor(idx);
        const size_t hi = (size_t) std::ceil(idx);
        if (lo == hi) {
            return sorted[lo];
        }
        const double t = idx - lo;
        return sorted[lo] * (1.0 - t) + sorted[hi] * t;
    };

    summary.p50 = percentile(0.50);
    summary.p90 = percentile(0.90);
    summary.p95 = percentile(0.95);
    summary.max = sorted.back();
    return summary;
}

static metric_summary summarize_metrics(const global_metrics & metrics) {
    metric_summary summary;
    const double denom = std::max(1, metrics.compared_positions);
    summary.compared_positions = metrics.compared_positions;
    summary.mean_topk_kl = metrics.sum_topk_kl / denom;
    summary.weighted_topk_kl = metrics.sum_weighted_topk_kl / std::max(1e-9, metrics.sum_margin_weight);
    summary.mean_topk_overlap = metrics.sum_topk_overlap / denom;
    summary.mean_rank_drift = metrics.sum_rank_drift / denom;
    summary.teacher_margin_mean = metrics.sum_teacher_margin / denom;
    summary.mean_next_logprob_delta = metrics.sum_next_logprob_delta / denom;
    summary.mean_next_rank_drift = metrics.sum_next_rank_drift / denom;
    summary.teacher_winner_in_student_topk = metrics.sum_teacher_winner_in_student_topk / denom;
    summary.next_token_in_teacher_topk = metrics.sum_next_token_in_teacher_topk / denom;
    summary.next_token_in_student_topk = metrics.sum_next_token_in_student_topk / denom;
    summary.argmax_flip_rate = (double) metrics.argmax_flips / denom;
    summary.topk_kl_percentiles = summarize_percentiles(metrics.topk_kl_values);
    summary.rank_drift_percentiles = summarize_percentiles(metrics.rank_drift_values);
    summary.teacher_margin_percentiles = summarize_percentiles(metrics.teacher_margin_values);
    summary.next_logprob_delta_percentiles = summarize_percentiles(metrics.next_logprob_delta_values);
    return summary;
}

static metric_delta compute_metric_delta(const metric_summary & baseline, const metric_summary & candidate) {
    metric_delta delta;
    delta.delta_mean_topk_kl = candidate.mean_topk_kl - baseline.mean_topk_kl;
    delta.delta_weighted_topk_kl = candidate.weighted_topk_kl - baseline.weighted_topk_kl;
    delta.delta_mean_topk_overlap = candidate.mean_topk_overlap - baseline.mean_topk_overlap;
    delta.delta_mean_rank_drift = candidate.mean_rank_drift - baseline.mean_rank_drift;
    delta.delta_mean_next_logprob_delta = candidate.mean_next_logprob_delta - baseline.mean_next_logprob_delta;
    delta.delta_mean_next_rank_drift = candidate.mean_next_rank_drift - baseline.mean_next_rank_drift;
    delta.delta_teacher_winner_in_student_topk = candidate.teacher_winner_in_student_topk - baseline.teacher_winner_in_student_topk;
    delta.delta_next_token_in_student_topk = candidate.next_token_in_student_topk - baseline.next_token_in_student_topk;
    delta.delta_argmax_flip_rate = candidate.argmax_flip_rate - baseline.argmax_flip_rate;
    delta.delta_p95_topk_kl = candidate.topk_kl_percentiles.p95 - baseline.topk_kl_percentiles.p95;
    delta.delta_p95_rank_drift = candidate.rank_drift_percentiles.p95 - baseline.rank_drift_percentiles.p95;

    // Positive means the candidate looks worse than the baseline.
    delta.damage_score =
            delta.delta_mean_topk_kl
          + delta.delta_weighted_topk_kl
          + delta.delta_p95_topk_kl
          + 0.25 * delta.delta_argmax_flip_rate
          + 0.01 * delta.delta_mean_rank_drift
          + 0.01 * delta.delta_mean_next_rank_drift
          - delta.delta_mean_next_logprob_delta
          - 0.25 * delta.delta_mean_topk_overlap
          - 0.25 * delta.delta_next_token_in_student_topk;
    return delta;
}

static topk_stats compute_topk_stats(const float * logits, int n_vocab, int top_k) {
    topk_stats stats;
    using heap_entry = std::pair<float, llama_token>;
    auto cmp = [] (const heap_entry & a, const heap_entry & b) {
        return a.first > b.first;
    };
    std::vector<heap_entry> heap;
    heap.reserve(top_k);

    for (int i = 0; i < n_vocab; ++i) {
        const float logit = logits[i];
        if (logit > stats.max_logit) {
            stats.second_logit = stats.max_logit;
            stats.max_logit = logit;
            stats.argmax = i;
        } else if (logit > stats.second_logit) {
            stats.second_logit = logit;
        }

        if ((int) heap.size() < top_k) {
            heap.emplace_back(logit, i);
            std::push_heap(heap.begin(), heap.end(), cmp);
        } else if (logit > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            heap.back() = {logit, i};
            std::push_heap(heap.begin(), heap.end(), cmp);
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp((double) logits[i] - stats.max_logit);
    }
    stats.log_sum_exp = std::log(sum_exp) + stats.max_logit;

    std::sort(heap.begin(), heap.end(), [] (const heap_entry & a, const heap_entry & b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });

    stats.entries.reserve(heap.size());
    for (const auto & entry : heap) {
        stats.entries.push_back({entry.second, entry.first});
    }
    return stats;
}

static double compute_topk_kl(
        const topk_stats & teacher_topk,
        const float * student_logits,
        int top_k) {
    const int k = std::min<int>(top_k, teacher_topk.entries.size());
    if (k <= 0) {
        return 0.0;
    }

    std::vector<double> teacher_probs(k);
    std::vector<double> student_probs(k);

    double teacher_sum = 0.0;
    double student_sum = 0.0;
    for (int i = 0; i < k; ++i) {
        const auto & entry = teacher_topk.entries[i];
        teacher_probs[i] = std::exp((double) entry.logit - teacher_topk.max_logit);
        teacher_sum += teacher_probs[i];

        const float student_logit = student_logits[entry.token];
        student_probs[i] = std::exp((double) student_logit - teacher_topk.max_logit);
        student_sum += student_probs[i];
    }

    if (teacher_sum <= 0.0 || student_sum <= 0.0) {
        return 0.0;
    }

    double kl = 0.0;
    for (int i = 0; i < k; ++i) {
        const double p = teacher_probs[i] / teacher_sum;
        const double q = std::max(student_probs[i] / student_sum, 1e-12);
        kl += p * (std::log(std::max(p, 1e-12)) - std::log(q));
    }
    return kl;
}

static double compute_topk_overlap(
        const topk_stats & teacher_topk,
        const topk_stats & student_topk,
        int top_k) {
    const int k_teacher = std::min<int>(top_k, teacher_topk.entries.size());
    const int k_student = std::min<int>(top_k, student_topk.entries.size());
    if (k_teacher <= 0 || k_student <= 0) {
        return 0.0;
    }

    std::set<llama_token> student_ids;
    for (int i = 0; i < k_student; ++i) {
        student_ids.insert(student_topk.entries[i].token);
    }

    int overlap = 0;
    for (int i = 0; i < k_teacher; ++i) {
        overlap += student_ids.count(teacher_topk.entries[i].token) != 0;
    }
    return (double) overlap / k_teacher;
}

static double compute_rank_drift(
        const topk_stats & teacher_topk,
        const topk_stats & student_topk,
        int top_k) {
    const int k = std::min<int>(top_k, teacher_topk.entries.size());
    if (k <= 0) {
        return 0.0;
    }

    std::unordered_map<llama_token, int> student_rank;
    student_rank.reserve(student_topk.entries.size());
    for (int i = 0; i < (int) student_topk.entries.size(); ++i) {
        student_rank.emplace(student_topk.entries[i].token, i + 1);
    }

    double drift = 0.0;
    for (int i = 0; i < k; ++i) {
        const auto it = student_rank.find(teacher_topk.entries[i].token);
        const int rank_student = it != student_rank.end() ? it->second : top_k + 1;
        drift += std::abs(rank_student - (i + 1));
    }
    return drift / k;
}

static double compute_teacher_margin(const topk_stats & teacher_topk) {
    if (teacher_topk.entries.size() < 2) {
        return 0.0;
    }
    const double p1 = std::exp((double) teacher_topk.entries[0].logit - teacher_topk.log_sum_exp);
    const double p2 = std::exp((double) teacher_topk.entries[1].logit - teacher_topk.log_sum_exp);
    return p1 - p2;
}

static int compute_token_rank(const float * logits, int n_vocab, llama_token token) {
    const float target = logits[token];
    int rank = 1;
    for (int i = 0; i < n_vocab; ++i) {
        if (i == token) {
            continue;
        }
        if (logits[i] > target || (logits[i] == target && i < token)) {
            ++rank;
        }
    }
    return rank;
}

static bool topk_contains_token(const topk_stats & stats, llama_token token, int top_k) {
    const int k = std::min<int>(top_k, stats.entries.size());
    for (int i = 0; i < k; ++i) {
        if (stats.entries[i].token == token) {
            return true;
        }
    }
    return false;
}

static double compute_logprob_for_token(const topk_stats & stats, llama_token token, const float * logits) {
    return (double) logits[token] - stats.log_sum_exp;
}

static per_position_metrics compare_position(
        const topk_stats & teacher_topk,
        const topk_stats & student_topk,
        const float * teacher_ptr,
        const float * student_ptr,
        int n_vocab,
        int top_k,
        llama_token next_token) {
    per_position_metrics metrics;
    metrics.topk_kl = compute_topk_kl(teacher_topk, student_ptr, top_k);
    metrics.topk_overlap = compute_topk_overlap(teacher_topk, student_topk, top_k);
    metrics.rank_drift = compute_rank_drift(teacher_topk, student_topk, top_k);
    metrics.teacher_margin = compute_teacher_margin(teacher_topk);
    metrics.margin_weight = 1.0 / std::max(metrics.teacher_margin, 1e-3);
    metrics.argmax_flip = teacher_topk.argmax != student_topk.argmax;
    metrics.teacher_winner_in_student_topk = topk_contains_token(student_topk, teacher_topk.argmax, top_k);
    metrics.next_token_in_teacher_topk = topk_contains_token(teacher_topk, next_token, top_k);
    metrics.next_token_in_student_topk = topk_contains_token(student_topk, next_token, top_k);

    const double teacher_next_logprob = compute_logprob_for_token(teacher_topk, next_token, teacher_ptr);
    const double student_next_logprob = compute_logprob_for_token(student_topk, next_token, student_ptr);
    metrics.next_logprob_delta = student_next_logprob - teacher_next_logprob;

    const int teacher_next_rank = compute_token_rank(teacher_ptr, n_vocab, next_token);
    const int student_next_rank = compute_token_rank(student_ptr, n_vocab, next_token);
    metrics.next_rank_drift = std::abs(student_next_rank - teacher_next_rank);
    return metrics;
}

static std::vector<float> decode_logits_for_chunk(
        llama_context * ctx,
        const llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
        int start,
        int n_ctx,
        int n_batch,
        int first) {
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int end = start + n_ctx;
    const int num_batches = (n_ctx + n_batch - 1) / n_batch;

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(std::min(n_batch, n_ctx), 0, 1);
    std::vector<float> logits;
    logits.reserve(size_t(n_ctx - first) * llama_vocab_n_tokens(vocab));

    for (int j = 0; j < num_batches; ++j) {
        const int batch_start = start + j * n_batch;
        const int batch_size = std::min(end - batch_start, n_batch);
        int n_outputs = 0;

        batch.n_tokens = 0;
        for (int k = 0; k < batch_size; ++k) {
            const int pos = j * n_batch + k;
            batch.token[k] = tokens[batch_start + k];
            if (add_bos && j == 0 && k == 0) {
                batch.token[k] = llama_vocab_bos(vocab);
            }
            batch.pos[k] = pos;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = pos >= first ? 1 : 0;
            n_outputs += batch.logits[k] != 0;
        }
        batch.n_tokens = batch_size;

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("failed to decode batch");
        }

        if (n_outputs > 0) {
            const float * batch_logits = llama_get_logits(ctx);
            const size_t n_vocab = llama_vocab_n_tokens(vocab);
            logits.insert(logits.end(), batch_logits, batch_logits + size_t(n_outputs) * n_vocab);
        }
    }

    llama_batch_free(batch);
    return logits;
}

static compare_result compare_models(
        llama_context * teacher_ctx,
        llama_context * student_ctx,
        const llama_vocab * teacher_vocab,
        const llama_vocab * student_vocab,
        const std::vector<llama_token> & tokens,
        int n_ctx,
        int n_batch,
        int n_chunks_limit,
        int top_k,
        attribution_capture * teacher_capture,
        attribution_capture * student_capture) {
    compare_result result;
    global_metrics & global = result.metrics;
    const int first = n_ctx / 2;
    const int n_vocab = llama_vocab_n_tokens(teacher_vocab);
    const int n_chunk = (tokens.size() - 1) / n_ctx;
    const int n_chunk_eval = n_chunks_limit > 0 ? std::min(n_chunk, n_chunks_limit) : n_chunk;

    LOG_INF("%s: comparing over %d chunks, n_ctx=%d, batch_size=%d, first_eval=%d, top_k=%d\n",
            __func__, n_chunk_eval, n_ctx, n_batch, first, top_k);

    for (int i = 0; i < n_chunk_eval; ++i) {
        const int start = i * n_ctx;
        if (teacher_capture) {
            teacher_capture->samples.clear();
        }
        if (student_capture) {
            student_capture->samples.clear();
        }
        std::vector<float> teacher_logits = decode_logits_for_chunk(teacher_ctx, teacher_vocab, tokens, start, n_ctx, n_batch, first);
        std::vector<float> student_logits = decode_logits_for_chunk(student_ctx, student_vocab, tokens, start, n_ctx, n_batch, first);

        const int available_positions = std::min<int>(teacher_logits.size(), student_logits.size()) / n_vocab;
        const int compare_positions = std::max(0, std::min(available_positions, n_ctx - first - 1));
        if (compare_positions == 0) {
            continue;
        }

        chunk_metrics chunk;
        chunk.chunk_index = i + 1;
        chunk.compared_positions = compare_positions;
        std::vector<double> chunk_topk_kl_values;
        std::vector<double> chunk_rank_drift_values;
        double chunk_margin_weight = 0.0;
        chunk_topk_kl_values.reserve(compare_positions);
        chunk_rank_drift_values.reserve(compare_positions);

        for (int j = 0; j < compare_positions; ++j) {
            const float * teacher_ptr = teacher_logits.data() + size_t(j) * n_vocab;
            const float * student_ptr = student_logits.data() + size_t(j) * n_vocab;
            const llama_token next_token = tokens[start + first + j + 1];

            const topk_stats teacher_topk = compute_topk_stats(teacher_ptr, n_vocab, top_k);
            const topk_stats student_topk = compute_topk_stats(student_ptr, n_vocab, top_k);
            const per_position_metrics pos = compare_position(
                    teacher_topk, student_topk, teacher_ptr, student_ptr, n_vocab, top_k, next_token);

            chunk.mean_topk_kl += pos.topk_kl;
            chunk.mean_topk_overlap += pos.topk_overlap;
            chunk.mean_rank_drift += pos.rank_drift;
            chunk.mean_teacher_margin += pos.teacher_margin;
            chunk.weighted_topk_kl += pos.topk_kl * pos.margin_weight;
            chunk_margin_weight += pos.margin_weight;
            chunk.mean_next_logprob_delta += pos.next_logprob_delta;
            chunk.mean_next_rank_drift += pos.next_rank_drift;
            chunk.teacher_winner_in_student_topk += pos.teacher_winner_in_student_topk ? 1.0 : 0.0;
            chunk.next_token_in_teacher_topk += pos.next_token_in_teacher_topk ? 1.0 : 0.0;
            chunk.next_token_in_student_topk += pos.next_token_in_student_topk ? 1.0 : 0.0;
            chunk.argmax_flips += pos.argmax_flip ? 1 : 0;

            chunk_topk_kl_values.push_back(pos.topk_kl);
            chunk_rank_drift_values.push_back(pos.rank_drift);
            global.topk_kl_values.push_back(pos.topk_kl);
            global.rank_drift_values.push_back(pos.rank_drift);
            global.teacher_margin_values.push_back(pos.teacher_margin);
            global.next_logprob_delta_values.push_back(pos.next_logprob_delta);
            global.sum_weighted_topk_kl += pos.topk_kl * pos.margin_weight;
            global.sum_margin_weight += pos.margin_weight;
            global.sum_next_logprob_delta += pos.next_logprob_delta;
            global.sum_next_rank_drift += pos.next_rank_drift;
            global.sum_teacher_winner_in_student_topk += pos.teacher_winner_in_student_topk ? 1.0 : 0.0;
            global.sum_next_token_in_teacher_topk += pos.next_token_in_teacher_topk ? 1.0 : 0.0;
            global.sum_next_token_in_student_topk += pos.next_token_in_student_topk ? 1.0 : 0.0;
        }

        chunk.mean_topk_kl /= compare_positions;
        chunk.mean_topk_overlap /= compare_positions;
        chunk.mean_rank_drift /= compare_positions;
        chunk.mean_teacher_margin /= compare_positions;
        chunk.weighted_topk_kl /= std::max(1e-9, chunk_margin_weight);
        chunk.mean_next_logprob_delta /= compare_positions;
        chunk.mean_next_rank_drift /= compare_positions;
        chunk.teacher_winner_in_student_topk /= compare_positions;
        chunk.next_token_in_teacher_topk /= compare_positions;
        chunk.next_token_in_student_topk /= compare_positions;

        const percentile_summary kl_summary = summarize_percentiles(chunk_topk_kl_values);
        chunk.p50_topk_kl = kl_summary.p50;
        chunk.p90_topk_kl = kl_summary.p90;
        chunk.p95_topk_kl = kl_summary.p95;
        chunk.max_topk_kl = kl_summary.max;

        const percentile_summary rank_summary = summarize_percentiles(chunk_rank_drift_values);
        chunk.p50_rank_drift = rank_summary.p50;
        chunk.p90_rank_drift = rank_summary.p90;
        chunk.p95_rank_drift = rank_summary.p95;
        chunk.max_rank_drift = rank_summary.max;

        global.compared_positions += compare_positions;
        global.argmax_flips += chunk.argmax_flips;
        global.sum_topk_kl += chunk.mean_topk_kl * compare_positions;
        global.sum_topk_overlap += chunk.mean_topk_overlap * compare_positions;
        global.sum_rank_drift += chunk.mean_rank_drift * compare_positions;
        global.sum_teacher_margin += chunk.mean_teacher_margin * compare_positions;
        global.chunks.push_back(chunk);

        LOG_INF("%s: chunk=%3d positions=%4d topk_kl=%10.6g overlap=%7.4f rank_drift=%7.4f argmax_flip_rate=%7.4f teacher_margin=%8.6f weighted_kl=%10.6g next_logprob_delta=%9.6f next_rank_drift=%7.4f\n",
                __func__, chunk.chunk_index, chunk.compared_positions,
                chunk.mean_topk_kl, chunk.mean_topk_overlap, chunk.mean_rank_drift,
                chunk.compared_positions > 0 ? (double) chunk.argmax_flips / chunk.compared_positions : 0.0,
                chunk.mean_teacher_margin, chunk.weighted_topk_kl, chunk.mean_next_logprob_delta, chunk.mean_next_rank_drift);

        if (teacher_capture && student_capture) {
            accumulate_attribution(result.attribution, *teacher_capture, *student_capture);
        }
    }

    return result;
}

static void write_json_summary(
        const std::string & path,
        const logit_compare_args & lc_args,
        const common_params & params,
        const compare_result & candidate_result,
        const compare_result * baseline_result) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open json output file");
    }

    const global_metrics & candidate_metrics = candidate_result.metrics;
    const global_metrics * baseline_metrics = baseline_result ? &baseline_result->metrics : nullptr;
    const metric_summary candidate = summarize_metrics(candidate_metrics);
    const metric_summary baseline = baseline_metrics ? summarize_metrics(*baseline_metrics) : metric_summary{};
    const metric_delta delta = baseline_metrics ? compute_metric_delta(baseline, candidate) : metric_delta{};
    const auto candidate_tensor_attr = summarize_attribution(candidate_result.attribution.tensors);
    const auto candidate_family_attr = summarize_attribution(candidate_result.attribution.families);
    const auto candidate_layer_attr = summarize_attribution(candidate_result.attribution.layers);
    const auto baseline_tensor_attr = baseline_result ? summarize_attribution(baseline_result->attribution.tensors) : std::vector<attribution_group_summary>{};
    const auto baseline_family_attr = baseline_result ? summarize_attribution(baseline_result->attribution.families) : std::vector<attribution_group_summary>{};
    const auto baseline_layer_attr = baseline_result ? summarize_attribution(baseline_result->attribution.layers) : std::vector<attribution_group_summary>{};
    const auto tensor_delta_attr = baseline_result ? compute_attribution_delta(baseline_tensor_attr, candidate_tensor_attr) : std::vector<attribution_delta_summary>{};
    const auto family_delta_attr = baseline_result ? compute_attribution_delta(baseline_family_attr, candidate_family_attr) : std::vector<attribution_delta_summary>{};
    const auto layer_delta_attr = baseline_result ? compute_attribution_delta(baseline_layer_attr, candidate_layer_attr) : std::vector<attribution_delta_summary>{};

    auto write_percentiles = [&out](const char * name, const percentile_summary & p, const char * suffix) {
        out << "    \"" << name << "\": {\n";
        out << "      \"p50\": " << p.p50 << ",\n";
        out << "      \"p90\": " << p.p90 << ",\n";
        out << "      \"p95\": " << p.p95 << ",\n";
        out << "      \"max\": " << p.max << "\n";
        out << "    }" << suffix << "\n";
    };

    auto write_summary = [&out, &write_percentiles](const char * name, const metric_summary & summary, const char * suffix) {
        out << "  \"" << name << "\": {\n";
        out << "    \"compared_positions\": " << summary.compared_positions << ",\n";
        out << "    \"mean_topk_kl\": " << summary.mean_topk_kl << ",\n";
        out << "    \"weighted_topk_kl\": " << summary.weighted_topk_kl << ",\n";
        out << "    \"mean_topk_overlap\": " << summary.mean_topk_overlap << ",\n";
        out << "    \"mean_rank_drift\": " << summary.mean_rank_drift << ",\n";
        out << "    \"teacher_margin_mean\": " << summary.teacher_margin_mean << ",\n";
        out << "    \"mean_next_logprob_delta\": " << summary.mean_next_logprob_delta << ",\n";
        out << "    \"mean_next_rank_drift\": " << summary.mean_next_rank_drift << ",\n";
        out << "    \"teacher_winner_in_student_topk\": " << summary.teacher_winner_in_student_topk << ",\n";
        out << "    \"next_token_in_teacher_topk\": " << summary.next_token_in_teacher_topk << ",\n";
        out << "    \"next_token_in_student_topk\": " << summary.next_token_in_student_topk << ",\n";
        out << "    \"argmax_flip_rate\": " << summary.argmax_flip_rate << ",\n";
        write_percentiles("topk_kl_percentiles", summary.topk_kl_percentiles, ",");
        write_percentiles("rank_drift_percentiles", summary.rank_drift_percentiles, ",");
        write_percentiles("teacher_margin_percentiles", summary.teacher_margin_percentiles, ",");
        write_percentiles("next_logprob_delta_percentiles", summary.next_logprob_delta_percentiles, "");
        out << "  }" << suffix << "\n";
    };

    auto write_chunks = [&out](const char * name, const global_metrics & metrics, const char * suffix) {
        out << "  \"" << name << "\": [\n";
        for (size_t i = 0; i < metrics.chunks.size(); ++i) {
            const chunk_metrics & chunk = metrics.chunks[i];
            out << "    {\n";
            out << "      \"chunk\": " << chunk.chunk_index << ",\n";
            out << "      \"positions\": " << chunk.compared_positions << ",\n";
            out << "      \"mean_topk_kl\": " << chunk.mean_topk_kl << ",\n";
            out << "      \"weighted_topk_kl\": " << chunk.weighted_topk_kl << ",\n";
            out << "      \"mean_topk_overlap\": " << chunk.mean_topk_overlap << ",\n";
            out << "      \"mean_rank_drift\": " << chunk.mean_rank_drift << ",\n";
            out << "      \"teacher_margin_mean\": " << chunk.mean_teacher_margin << ",\n";
            out << "      \"mean_next_logprob_delta\": " << chunk.mean_next_logprob_delta << ",\n";
            out << "      \"mean_next_rank_drift\": " << chunk.mean_next_rank_drift << ",\n";
            out << "      \"teacher_winner_in_student_topk\": " << chunk.teacher_winner_in_student_topk << ",\n";
            out << "      \"next_token_in_teacher_topk\": " << chunk.next_token_in_teacher_topk << ",\n";
            out << "      \"next_token_in_student_topk\": " << chunk.next_token_in_student_topk << ",\n";
            out << "      \"p50_topk_kl\": " << chunk.p50_topk_kl << ",\n";
            out << "      \"p90_topk_kl\": " << chunk.p90_topk_kl << ",\n";
            out << "      \"p95_topk_kl\": " << chunk.p95_topk_kl << ",\n";
            out << "      \"max_topk_kl\": " << chunk.max_topk_kl << ",\n";
            out << "      \"p50_rank_drift\": " << chunk.p50_rank_drift << ",\n";
            out << "      \"p90_rank_drift\": " << chunk.p90_rank_drift << ",\n";
            out << "      \"p95_rank_drift\": " << chunk.p95_rank_drift << ",\n";
            out << "      \"max_rank_drift\": " << chunk.max_rank_drift << ",\n";
            out << "      \"argmax_flip_rate\": " << (chunk.compared_positions > 0 ? (double) chunk.argmax_flips / chunk.compared_positions : 0.0) << "\n";
            out << "    }" << (i + 1 == metrics.chunks.size() ? "\n" : ",\n");
        }
        out << "  ]" << suffix << "\n";
    };

    auto write_attr_groups = [&out](const char * name, const std::vector<attribution_group_summary> & items, const char * suffix) {
        out << "  \"" << name << "\": [\n";
        for (size_t i = 0; i < items.size(); ++i) {
            const auto & item = items[i];
            out << "    {\n";
            out << "      \"key\": \"" << json_escape(item.key) << "\",\n";
            out << "      \"layer\": " << item.layer << ",\n";
            out << "      \"compared_tensors\": " << item.compared_tensors << ",\n";
            out << "      \"compared_values\": " << item.compared_values << ",\n";
            out << "      \"mean_mse\": " << item.mean_mse << ",\n";
            out << "      \"mean_cosine_error\": " << item.mean_cosine_error << ",\n";
            out << "      \"mean_norm_ratio_error\": " << item.mean_norm_ratio_error << ",\n";
            out << "      \"mean_abs_error\": " << item.mean_abs_error << ",\n";
            out << "      \"max_mse\": " << item.max_mse << "\n";
            out << "    }" << (i + 1 == items.size() ? "\n" : ",\n");
        }
        out << "  ]" << suffix << "\n";
    };

    auto write_attr_deltas = [&out](const char * name, const std::vector<attribution_delta_summary> & items, const char * suffix) {
        out << "  \"" << name << "\": [\n";
        for (size_t i = 0; i < items.size(); ++i) {
            const auto & item = items[i];
            out << "    {\n";
            out << "      \"key\": \"" << json_escape(item.key) << "\",\n";
            out << "      \"layer\": " << item.layer << ",\n";
            out << "      \"delta_mean_mse\": " << item.delta_mean_mse << ",\n";
            out << "      \"delta_mean_cosine_error\": " << item.delta_mean_cosine_error << ",\n";
            out << "      \"delta_mean_norm_ratio_error\": " << item.delta_mean_norm_ratio_error << ",\n";
            out << "      \"delta_mean_abs_error\": " << item.delta_mean_abs_error << ",\n";
            out << "      \"delta_max_mse\": " << item.delta_max_mse << "\n";
            out << "    }" << (i + 1 == items.size() ? "\n" : ",\n");
        }
        out << "  ]" << suffix << "\n";
    };

    out << "{\n";
    out << "  \"teacher\": \"" << json_escape(lc_args.teacher_model) << "\",\n";
    if (!lc_args.baseline_model.empty()) {
        out << "  \"baseline\": \"" << json_escape(lc_args.baseline_model) << "\",\n";
    }
    out << "  \"student\": \"" << json_escape(lc_args.student_model) << "\",\n";
    out << "  \"file\": \"" << json_escape(params.prompt_file) << "\",\n";
    out << "  \"top_k\": " << lc_args.top_k << ",\n";
    out << "  \"n_ctx\": " << params.n_ctx << ",\n";
    out << "  \"layer_attribution\": " << (lc_args.layer_attribution ? "true" : "false") << ",\n";
    out << "  \"layer_sample_tokens\": " << lc_args.layer_sample_tokens << ",\n";

    if (baseline_metrics) {
        write_summary("baseline_summary", baseline, ",");
    }
    write_summary("candidate_summary", candidate, ",");

    if (baseline_metrics) {
        out << "  \"delta\": {\n";
        out << "    \"delta_mean_topk_kl\": " << delta.delta_mean_topk_kl << ",\n";
        out << "    \"delta_weighted_topk_kl\": " << delta.delta_weighted_topk_kl << ",\n";
        out << "    \"delta_mean_topk_overlap\": " << delta.delta_mean_topk_overlap << ",\n";
        out << "    \"delta_mean_rank_drift\": " << delta.delta_mean_rank_drift << ",\n";
        out << "    \"delta_mean_next_logprob_delta\": " << delta.delta_mean_next_logprob_delta << ",\n";
        out << "    \"delta_mean_next_rank_drift\": " << delta.delta_mean_next_rank_drift << ",\n";
        out << "    \"delta_teacher_winner_in_student_topk\": " << delta.delta_teacher_winner_in_student_topk << ",\n";
        out << "    \"delta_next_token_in_student_topk\": " << delta.delta_next_token_in_student_topk << ",\n";
        out << "    \"delta_argmax_flip_rate\": " << delta.delta_argmax_flip_rate << ",\n";
        out << "    \"delta_p95_topk_kl\": " << delta.delta_p95_topk_kl << ",\n";
        out << "    \"delta_p95_rank_drift\": " << delta.delta_p95_rank_drift << ",\n";
        out << "    \"damage_score\": " << delta.damage_score << "\n";
        out << "  },\n";
        write_chunks("baseline_chunks", *baseline_metrics, ",");
    }
    write_chunks("candidate_chunks", candidate_metrics, ",");

    if (baseline_result) {
        write_attr_groups("baseline_tensor_attribution", baseline_tensor_attr, ",");
        write_attr_groups("baseline_family_attribution", baseline_family_attr, ",");
        write_attr_groups("baseline_layer_attribution", baseline_layer_attr, ",");
    }
    write_attr_groups("candidate_tensor_attribution", candidate_tensor_attr, ",");
    write_attr_groups("candidate_family_attribution", candidate_family_attr, ",");
    if (baseline_result) {
        write_attr_groups("candidate_layer_attribution", candidate_layer_attr, ",");
        write_attr_deltas("tensor_attribution_delta", tensor_delta_attr, ",");
        write_attr_deltas("family_attribution_delta", family_delta_attr, ",");
        write_attr_deltas("layer_attribution_delta", layer_delta_attr, "");
    } else {
        write_attr_groups("candidate_layer_attribution", candidate_layer_attr, "");
    }
    out << "}\n";
}

static void log_metric_summary(const char * label, const metric_summary & summary) {
    LOG_INF("%s: positions=%d topk_kl=%10.6g weighted_kl=%10.6g overlap=%7.4f rank_drift=%7.4f flips=%7.4f kl_p95=%10.6g rank_p95=%7.4f\n",
            label, summary.compared_positions,
            summary.mean_topk_kl,
            summary.weighted_topk_kl,
            summary.mean_topk_overlap,
            summary.mean_rank_drift,
            summary.argmax_flip_rate,
            summary.topk_kl_percentiles.p95,
            summary.rank_drift_percentiles.p95);
    LOG_INF("%s: teacher_margin=%8.6f next_logprob_delta=%9.6f next_rank_drift=%7.4f teacher_winner_topk=%7.4f next_in_student_topk=%7.4f\n",
            label,
            summary.teacher_margin_mean,
            summary.mean_next_logprob_delta,
            summary.mean_next_rank_drift,
            summary.teacher_winner_in_student_topk,
            summary.next_token_in_student_topk);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_init();

    logit_compare_args lc_args;
    std::vector<std::string> passthrough;
    if (!parse_logit_compare_args(argc, argv, lc_args, passthrough)) {
        print_usage(argc, argv);
        return 1;
    }

    std::vector<char *> filtered_argv = make_argv(passthrough);

    common_params params;
    params.escape = false;
    params.kl_divergence = false;
    params.ppl_output_type = 0;

    if (!common_params_parse((int) filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_PERPLEXITY, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("%s: an evaluation file or prompt is required\n", __func__);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    attribution_capture teacher_capture;
    teacher_capture.enabled = lc_args.layer_attribution;
    teacher_capture.sample_tokens = lc_args.layer_sample_tokens;

    common_params teacher_params = params;
    teacher_params.model.path = lc_args.teacher_model;
    if (lc_args.layer_attribution) {
        teacher_params.cb_eval = capture_attribution_cb;
        teacher_params.cb_eval_user_data = &teacher_capture;
    }

    common_init_result_ptr teacher_init = common_init_from_params(teacher_params);

    llama_model * teacher_model = teacher_init->model();
    llama_context * teacher_ctx = teacher_init->context();

    if (teacher_model == nullptr || teacher_ctx == nullptr) {
        LOG_ERR("%s: failed to load teacher model\n", __func__);
        return 1;
    }

    const llama_vocab * teacher_vocab = llama_model_get_vocab(teacher_model);
    const int teacher_n_vocab = llama_vocab_n_tokens(teacher_vocab);

    std::vector<llama_token> teacher_tokens = common_tokenize(teacher_ctx, params.prompt, true);
    if (teacher_tokens.size() < 2) {
        LOG_ERR("%s: not enough tokens to compare\n", __func__);
        return 1;
    }

    logit_log_section("Logit compare setup");
    LOG_INF("%s: teacher=%s\n", __func__, lc_args.teacher_model.c_str());
    LOG_INF("%s: tokens=%zu requested_chunks=%d top_k=%d\n",
            __func__, teacher_tokens.size(), params.n_chunks, lc_args.top_k);
    if (lc_args.layer_attribution) {
        LOG_INF("%s: layer attribution enabled sample_tokens=%d\n",
                __func__, lc_args.layer_sample_tokens);
    }

    auto compare_student = [&](const std::string & model_path, const char * label) -> compare_result {
        common_params student_params = params;
        student_params.model.path = model_path;
        attribution_capture student_capture;
        student_capture.enabled = lc_args.layer_attribution;
        student_capture.sample_tokens = lc_args.layer_sample_tokens;
        if (lc_args.layer_attribution) {
            student_params.cb_eval = capture_attribution_cb;
            student_params.cb_eval_user_data = &student_capture;
        }
        common_init_result_ptr student_init = common_init_from_params(student_params);

        llama_model * student_model = student_init->model();
        llama_context * student_ctx = student_init->context();
        if (student_model == nullptr || student_ctx == nullptr) {
            throw std::runtime_error(std::string("failed to load ") + label + " model");
        }

        const llama_vocab * student_vocab = llama_model_get_vocab(student_model);
        const int student_n_vocab = llama_vocab_n_tokens(student_vocab);
        if (teacher_n_vocab != student_n_vocab) {
            throw std::runtime_error(std::string("vocab mismatch for ") + label);
        }

        const std::vector<llama_token> student_tokens = common_tokenize(student_ctx, params.prompt, true);
        if (teacher_tokens != student_tokens) {
            throw std::runtime_error(std::string("tokenization mismatch between teacher and ") + label);
        }

        const int n_ctx = std::min(llama_n_ctx(teacher_ctx), llama_n_ctx(student_ctx));
        const int n_batch = std::min<int>(params.n_batch, n_ctx);
        const int n_chunk = (teacher_tokens.size() - 1) / n_ctx;
        if (n_chunk <= 0) {
            throw std::runtime_error("input does not contain enough tokens for one comparison chunk");
        }

        logit_log_section(label);
        LOG_INF("%s: %s=%s\n", __func__, label, model_path.c_str());
        LOG_INF("%s: %s n_ctx=%d n_batch=%d available_chunks=%d\n",
                __func__, label, n_ctx, n_batch, n_chunk);

        compare_result result = compare_models(
                teacher_ctx, student_ctx,
                teacher_vocab, student_vocab,
                teacher_tokens, n_ctx, n_batch, params.n_chunks, lc_args.top_k,
                lc_args.layer_attribution ? &teacher_capture : nullptr,
                lc_args.layer_attribution ? &student_capture : nullptr);

        if (result.metrics.compared_positions == 0) {
            throw std::runtime_error(std::string("no comparable positions were produced for ") + label);
        }
        return result;
    };

    compare_result candidate_result;
    compare_result baseline_result;
    const compare_result * baseline_result_ptr = nullptr;

    try {
        if (!lc_args.baseline_model.empty()) {
            baseline_result = compare_student(lc_args.baseline_model, "baseline");
            baseline_result_ptr = &baseline_result;
        }
        candidate_result = compare_student(lc_args.student_model, "candidate");
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }

    const metric_summary candidate_summary = summarize_metrics(candidate_result.metrics);
    logit_log_section("Metric summary");
    log_metric_summary("candidate_summary", candidate_summary);

    if (baseline_result_ptr != nullptr) {
        const metric_summary baseline_summary = summarize_metrics(baseline_result_ptr->metrics);
        log_metric_summary("baseline_summary", baseline_summary);
        if (baseline_summary.compared_positions != candidate_summary.compared_positions) {
            LOG_WRN("%s: baseline/candidate compared different position counts (%d vs %d); delta is still reported but should be treated cautiously\n",
                    __func__, baseline_summary.compared_positions, candidate_summary.compared_positions);
        }
        const metric_delta delta = compute_metric_delta(baseline_summary, candidate_summary);
        LOG_INF("%s: paired_delta topk_kl=%+10.6g weighted_kl=%+10.6g overlap=%+9.6f flips=%+9.6f next_logprob=%+9.6f kl_p95=%+10.6g damage=%+10.6g\n",
                __func__,
                delta.delta_mean_topk_kl,
                delta.delta_weighted_topk_kl,
                delta.delta_mean_topk_overlap,
                delta.delta_argmax_flip_rate,
                delta.delta_mean_next_logprob_delta,
                delta.delta_p95_topk_kl,
                delta.damage_score);
    }

    if (lc_args.layer_attribution) {
        logit_log_section("Attribution summary");
        const auto candidate_layers = summarize_attribution(candidate_result.attribution.layers);
        const auto candidate_families = summarize_attribution(candidate_result.attribution.families);
        const auto candidate_tensors = summarize_attribution(candidate_result.attribution.tensors);

        auto log_top_attr = [](
                const char * label,
                const std::vector<attribution_group_summary> & items,
                size_t limit) {
            for (size_t i = 0; i < std::min(limit, items.size()); ++i) {
                const auto & item = items[i];
                LOG_INF("%s: rank=%zu key=%s layer=%d mean_mse=%10.6g cosine_error=%10.6g norm_ratio_error=%10.6g compared_tensors=%d compared_values=%d\n",
                        label, i + 1, item.key.c_str(), item.layer, item.mean_mse,
                        item.mean_cosine_error, item.mean_norm_ratio_error,
                        item.compared_tensors, item.compared_values);
            }
        };

        log_top_attr("candidate_layer_attr", candidate_layers, 8);
        log_top_attr("candidate_family_attr", candidate_families, 8);
        log_top_attr("candidate_tensor_attr", candidate_tensors, 8);

        if (baseline_result_ptr != nullptr) {
            const auto baseline_layers = summarize_attribution(baseline_result_ptr->attribution.layers);
            const auto baseline_families = summarize_attribution(baseline_result_ptr->attribution.families);
            const auto layer_deltas = compute_attribution_delta(baseline_layers, candidate_layers);
            const auto family_deltas = compute_attribution_delta(baseline_families, candidate_families);

            for (size_t i = 0; i < std::min<size_t>(8, layer_deltas.size()); ++i) {
                const auto & item = layer_deltas[i];
                LOG_INF("paired_layer_delta: rank=%zu layer=%d delta_mean_mse=%10.6g delta_cosine_error=%10.6g delta_norm_ratio_error=%10.6g\n",
                        i + 1, item.layer, item.delta_mean_mse, item.delta_mean_cosine_error, item.delta_mean_norm_ratio_error);
            }
            for (size_t i = 0; i < std::min<size_t>(8, family_deltas.size()); ++i) {
                const auto & item = family_deltas[i];
                LOG_INF("paired_family_delta: rank=%zu key=%s delta_mean_mse=%10.6g delta_cosine_error=%10.6g delta_norm_ratio_error=%10.6g\n",
                        i + 1, item.key.c_str(), item.delta_mean_mse, item.delta_mean_cosine_error, item.delta_mean_norm_ratio_error);
            }
        }
    }

    if (!lc_args.json_out.empty()) {
        logit_log_section("Output");
        write_json_summary(lc_args.json_out, lc_args, params, candidate_result, baseline_result_ptr);
        LOG_INF("%s: wrote json summary to %s\n", __func__, lc_args.json_out.c_str());
    }

    return 0;
}
