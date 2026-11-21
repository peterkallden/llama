#include "recirculation-contract.h"

#include <cassert>

int main() {
    assert(recirculation_supported_architecture("qwen2"));
    assert(recirculation_supported_architecture("qwen3"));
    assert(!recirculation_supported_architecture("qwen35"));
    assert(recirculation_supported_architecture("phi3"));
    assert(!recirculation_supported_architecture("gpt2"));
    recirculation_config config{"qwen3-test", 8, 3, 0.1f, 1.0f, 1};
    assert(recirculation_config_valid(config, 32, 4096));
    config.source_layer_input = config.destination_layer_input;
    assert(!recirculation_config_valid(config, 32, 4096));

    const float d[] = {3.0f, 4.0f};
    const float s[] = {0.0f, 2.0f};
    std::vector<float> mixed;
    float sn = 0, dn = 0;
    assert(recirculation_mix(d, s, 2, 0.1f, 1.0f, mixed, sn, dn));
    assert(sn == 2.0f && dn == 5.0f);
    assert(mixed[0] == 3.0f && mixed[1] == 4.5f);
    assert(recirculation_mix(d, s, 2, 0.0f, 1.0f, mixed, sn, dn));
    assert(mixed[0] == d[0] && mixed[1] == d[1]);
    const auto previous = mixed;
    const float huge[] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    assert(!recirculation_mix(huge, s, 2, 1.0f, 1.0f, mixed, sn, dn));
    assert(mixed == previous);
    const float invalid[] = {std::numeric_limits<float>::quiet_NaN(), 1.0f};
    assert(!recirculation_mix(d, invalid, 2, 0.1f, 1.0f, mixed, sn, dn));
    const float zero[] = {0.0f, 0.0f};
    assert(!recirculation_mix(d, zero, 2, 0.1f, 1.0f, mixed, sn, dn));

    std::vector<float> matched;
    assert(recirculation_rescale_norm({3.0f, 4.0f}, 10.0f, matched));
    assert(std::abs(std::hypot(matched[0], matched[1]) - 10.0f) < 1e-5f);
    std::vector<float> repeated;
    assert(recirculation_rescale_norm({3.0f, 4.0f}, 10.0f, repeated) && repeated == matched);
    const auto stable = matched;
    assert(!recirculation_rescale_norm({0.0f, 0.0f}, 1.0f, matched));
    assert(matched == stable);
    std::vector<size_t> pieces{2, 1, 3, 2};
    size_t selected = 99;
    assert(recirculation_token_for_span(pieces, 1, 4, selected) && selected == 2);
    assert(!recirculation_token_for_span(pieces, 0, 10, selected));
    assert(recirculation_token_log_probability({0.0f, 0.0f}, 0) < 0.0);
    assert(std::isnan(recirculation_token_log_probability({1.0f}, 2)));
    assert(recirculation_answer_margin(-1.0, -2.0) == 1.0);
    assert(std::isnan(recirculation_answer_margin(-1.0, std::numeric_limits<double>::infinity())));
    double total = 0;
    assert(recirculation_sum_log_probabilities({-0.2, -0.3, -0.5}, total) && std::abs(total + 1.0) < 1e-12);
    assert(!recirculation_sum_log_probabilities({-0.2, std::numeric_limits<double>::quiet_NaN()}, total));
}
