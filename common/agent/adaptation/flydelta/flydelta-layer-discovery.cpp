#include "agent/adaptation/flydelta/flydelta-layer-discovery.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool finite(float value) {
    return std::isfinite(value);
}

float norm(const std::vector<float> & values) {
    double squared = 0.0;
    for (const float value : values) squared += static_cast<double>(value) * value;
    return static_cast<float>(std::sqrt(squared));
}

float dot(const std::vector<float> & left, const std::vector<float> & right) {
    double value = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        value += static_cast<double>(left[index]) * right[index];
    }
    return static_cast<float>(value);
}

bool capture_identity_matches(
        const common_flydelta_hidden_state_capture & capture,
        const common_flydelta_hidden_state_capture & reference,
        std::string & error) {
    if (capture.model_profile_fingerprint != reference.model_profile_fingerprint ||
            capture.capture_layout_revision != reference.capture_layout_revision ||
            capture.layer_indices != reference.layer_indices ||
            capture.n_embd != reference.n_embd || capture.position != reference.position ||
            (capture.position == common_flydelta_capture_position::prompt_row &&
             capture.token_index != reference.token_index)) {
        error = "FlyDelta layer discovery captures are not aligned";
        return false;
    }
    return true;
}

bool layer_offset(
        const common_flydelta_hidden_state_capture & capture,
        size_t layer,
        size_t & offset) {
    offset = layer * static_cast<size_t>(capture.n_embd);
    return offset + capture.n_embd <= capture.values.size();
}

bool add_anchor(
        size_t index,
        const std::vector<common_flydelta_layer_score> & scores,
        uint32_t min_separation,
        std::vector<size_t> & selected,
        bool enforce_separation) {
    if (std::find(selected.begin(), selected.end(), index) != selected.end()) return false;
    if (enforce_separation) {
        for (const size_t existing : selected) {
            const uint32_t distance = scores[index].layer_index > scores[existing].layer_index
                ? scores[index].layer_index - scores[existing].layer_index
                : scores[existing].layer_index - scores[index].layer_index;
            if (distance < min_separation) return false;
        }
    }
    selected.push_back(index);
    return true;
}

} // namespace

bool common_flydelta_layer_discovery_config_validate(
        const common_flydelta_layer_discovery_config & config,
        std::string & error) {
    error.clear();
    if (config.schema_version != 1 || config.max_samples_per_class == 0 ||
            config.max_samples_per_class > 256 || config.max_layers == 0 ||
            config.max_layers > 256 || config.max_anchors == 0 || config.max_anchors > 32 ||
            !finite(config.variance_ridge) || config.variance_ridge <= 0.0f ||
            config.max_capture_bytes == 0) {
        error = "FlyDelta layer discovery configuration is invalid";
        return false;
    }
    return true;
}

bool common_flydelta_layer_discovery_result_validate(
        const common_flydelta_layer_discovery_result & result,
        const common_flydelta_layer_discovery_config & config,
        std::string & error) {
    error.clear();
    if (!common_flydelta_layer_discovery_config_validate(config, error) ||
            result.schema_version != 1 || result.positive_samples == 0 ||
            result.negative_samples == 0 || result.n_embd == 0 ||
            result.scores.empty() || result.scores.size() > config.max_layers ||
            result.anchor_layers.empty() || result.anchor_layers.size() > config.max_anchors ||
            !std::is_sorted(result.anchor_layers.begin(), result.anchor_layers.end())) {
        if (error.empty()) error = "FlyDelta layer discovery result is invalid";
        return false;
    }
    for (size_t index = 0; index < result.scores.size(); ++index) {
        const auto & score = result.scores[index];
        if ((index != 0 && result.scores[index - 1].layer_index >= score.layer_index) ||
                score.direction.size() != result.n_embd || !finite(score.separation) ||
                !finite(score.projected_variance) || !finite(score.fisher_score) ||
                !finite(score.slope) || score.separation < 0.0f ||
                score.projected_variance < 0.0f || score.fisher_score < 0.0f) {
            error = "FlyDelta layer discovery score is invalid";
            return false;
        }
        for (const float value : score.direction) {
            if (!finite(value)) {
                error = "FlyDelta layer discovery direction is invalid";
                return false;
            }
        }
    }
    for (const uint32_t layer : result.anchor_layers) {
        const auto found = std::find_if(result.scores.begin(), result.scores.end(),
            [&](const auto & score) { return score.layer_index == layer; });
        if (found == result.scores.end()) {
            error = "FlyDelta layer discovery anchor is not scored";
            return false;
        }
    }
    return true;
}

bool common_flydelta_discover_layer_regions(
        const std::vector<common_flydelta_hidden_state_capture> & positive,
        const std::vector<common_flydelta_hidden_state_capture> & negative,
        const common_flydelta_layer_discovery_config & config,
        common_flydelta_layer_discovery_result & result,
        std::string & error) {
    error.clear();
    result = {};
    if (!common_flydelta_layer_discovery_config_validate(config, error) ||
            positive.empty() || positive.size() > config.max_samples_per_class ||
            negative.empty() || negative.size() > config.max_samples_per_class) {
        if (error.empty()) error = "FlyDelta layer discovery sample bounds are invalid";
        return false;
    }

    const auto validate_capture_set = [&](
            const std::vector<common_flydelta_hidden_state_capture> & captures) {
        for (const auto & capture : captures) {
            if (!common_flydelta_hidden_state_capture_validate(
                    capture, config.max_capture_bytes, error)) return false;
            if (capture.layer_indices.empty() || capture.layer_indices.size() > config.max_layers) {
                error = "FlyDelta layer discovery layer count is invalid";
                return false;
            }
        }
        return true;
    };
    if (!validate_capture_set(positive) || !validate_capture_set(negative)) return false;
    const auto & reference = positive.front();
    if (!capture_identity_matches(reference, negative.front(), error)) return false;
    for (const auto & capture : positive) {
        if (!capture_identity_matches(capture, reference, error)) return false;
    }
    for (const auto & capture : negative) {
        if (!capture_identity_matches(capture, reference, error)) return false;
    }

    const size_t layers = reference.layer_indices.size();
    const size_t dimension = reference.n_embd;
    std::vector<std::vector<double>> positive_mean(
        layers, std::vector<double>(dimension, 0.0));
    std::vector<std::vector<double>> negative_mean(
        layers, std::vector<double>(dimension, 0.0));
    for (const auto & capture : positive) {
        for (size_t layer = 0; layer < layers; ++layer) {
            size_t offset = 0;
            if (!layer_offset(capture, layer, offset)) {
                error = "FlyDelta layer discovery positive capture is truncated";
                return false;
            }
            for (size_t value = 0; value < dimension; ++value) {
                positive_mean[layer][value] += capture.values[offset + value];
            }
        }
    }
    for (const auto & capture : negative) {
        for (size_t layer = 0; layer < layers; ++layer) {
            size_t offset = 0;
            if (!layer_offset(capture, layer, offset)) {
                error = "FlyDelta layer discovery negative capture is truncated";
                return false;
            }
            for (size_t value = 0; value < dimension; ++value) {
                negative_mean[layer][value] += capture.values[offset + value];
            }
        }
    }

    result.schema_version = 1;
    result.positive_samples = positive.size();
    result.negative_samples = negative.size();
    result.n_embd = reference.n_embd;
    result.scores.reserve(layers);
    for (size_t layer = 0; layer < layers; ++layer) {
        std::vector<float> direction(dimension);
        for (size_t value = 0; value < dimension; ++value) {
            direction[value] = static_cast<float>(
                positive_mean[layer][value] / static_cast<double>(positive.size()) -
                negative_mean[layer][value] / static_cast<double>(negative.size()));
        }
        const float separation = norm(direction);
        if (!finite(separation)) {
            error = "FlyDelta layer discovery separation is invalid";
            return false;
        }
        if (separation > std::numeric_limits<float>::epsilon()) {
            for (float & value : direction) value /= separation;
        }
        double positive_variance = 0.0;
        double negative_variance = 0.0;
        for (const auto & capture : positive) {
            size_t offset = 0;
            layer_offset(capture, layer, offset);
            double projection = 0.0;
            for (size_t value = 0; value < dimension; ++value) {
                projection += (static_cast<double>(capture.values[offset + value]) -
                    positive_mean[layer][value] / static_cast<double>(positive.size())) *
                    direction[value];
            }
            positive_variance += projection * projection;
        }
        for (const auto & capture : negative) {
            size_t offset = 0;
            layer_offset(capture, layer, offset);
            double projection = 0.0;
            for (size_t value = 0; value < dimension; ++value) {
                projection += (static_cast<double>(capture.values[offset + value]) -
                    negative_mean[layer][value] / static_cast<double>(negative.size())) *
                    direction[value];
            }
            negative_variance += projection * projection;
        }
        const double variance = positive_variance / static_cast<double>(positive.size()) +
            negative_variance / static_cast<double>(negative.size());
        const float projected_variance = static_cast<float>(variance);
        const float fisher = separation /
            std::sqrt(projected_variance + config.variance_ridge);
        common_flydelta_layer_score score;
        score.layer_index = reference.layer_indices[layer];
        score.separation = separation;
        score.projected_variance = projected_variance;
        score.fisher_score = fisher;
        score.direction = std::move(direction);
        result.scores.push_back(std::move(score));
    }
    for (size_t index = 0; index < result.scores.size(); ++index) {
        if (result.scores.size() == 1) {
            result.scores[index].slope = 0.0f;
        } else if (index == 0) {
            result.scores[index].slope = result.scores[1].fisher_score -
                result.scores[0].fisher_score;
        } else if (index + 1 == result.scores.size()) {
            result.scores[index].slope = result.scores[index].fisher_score -
                result.scores[index - 1].fisher_score;
        } else {
            result.scores[index].slope = (result.scores[index + 1].fisher_score -
                result.scores[index - 1].fisher_score) * 0.5f;
        }
    }

    std::vector<size_t> priority;
    const auto add_best = [&](auto predicate) {
        const auto found = std::max_element(result.scores.begin(), result.scores.end(), predicate);
        if (found != result.scores.end()) priority.push_back(
            static_cast<size_t>(std::distance(result.scores.begin(), found)));
    };
    priority.push_back(0);
    add_best([](const auto & left, const auto & right) {
        return left.slope < right.slope;
    });
    add_best([](const auto & left, const auto & right) {
        return left.separation < right.separation;
    });
    add_best([](const auto & left, const auto & right) {
        return left.fisher_score < right.fisher_score;
    });
    if (result.scores.size() > 1) {
        const auto found = std::max_element(result.scores.begin(), result.scores.end(),
            [](const auto & left, const auto & right) { return left.slope < right.slope; });
        if (found != result.scores.end()) {
            const size_t index = static_cast<size_t>(std::distance(result.scores.begin(), found));
            priority.push_back(std::min(index + 1, result.scores.size() - 1));
        }
    }
    priority.push_back(result.scores.size() - 1);
    std::vector<size_t> ranking(result.scores.size());
    for (size_t index = 0; index < ranking.size(); ++index) ranking[index] = index;
    std::sort(ranking.begin(), ranking.end(), [&](size_t left, size_t right) {
        return result.scores[left].fisher_score > result.scores[right].fisher_score;
    });
    priority.insert(priority.end(), ranking.begin(), ranking.end());

    std::vector<size_t> selected;
    for (const size_t index : priority) {
        if (selected.size() == config.max_anchors) break;
        add_anchor(index, result.scores, config.min_anchor_separation, selected, true);
    }
    if (selected.size() < config.max_anchors) {
        for (const size_t index : priority) {
            if (selected.size() == config.max_anchors) break;
            add_anchor(index, result.scores, config.min_anchor_separation, selected, false);
        }
    }
    std::sort(selected.begin(), selected.end(), [&](size_t left, size_t right) {
        return result.scores[left].layer_index < result.scores[right].layer_index;
    });
    for (const size_t index : selected) result.anchor_layers.push_back(result.scores[index].layer_index);
    return common_flydelta_layer_discovery_result_validate(result, config, error);
}
