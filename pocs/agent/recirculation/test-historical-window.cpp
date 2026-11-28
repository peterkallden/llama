#include "alternative-direction.h"
#include "selective-mask.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
    const auto random_mask = recirculation::selective_random_mask(8, 2, 101);
    assert(random_mask == recirculation::selective_random_mask(8, 2, 101));
    assert(std::accumulate(random_mask.begin(), random_mask.end(), 0) == 2);
    const auto magnitude_mask = recirculation::selective_magnitude_mask({0, 5, 5, 1}, 2);
    assert(magnitude_mask == std::vector<uint8_t>({0, 1, 1, 0}));
    recirculation::experiment_case item;
    item.prompt_tokens.assign(20, 1);
    item.target_position = 5;
    const auto bounds = recirculation::historical_window_for_update(item);
    assert(bounds.start == 3 && bounds.target_offset == 2 && bounds.size == 6);
    assert(bounds.start + bounds.target_offset == size_t(item.target_position));
    assert(bounds.start + bounds.size <= item.prompt_tokens.size());
    item.target_position = 1;
    bool rejected = false;
    try { (void) recirculation::historical_window_for_update(item); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    item.target_position = 17;
    rejected = false;
    try { (void) recirculation::historical_window_for_update(item); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    item.task_type = "object_location";
    item.expected = "The pouch.";
    item.incorrect = "The drawer.";
    item.user = "The coin was kept in the drawer. UPDATE: A courier transfers it to the pouch. Where is the coin now?";
    assert(recirculation::alternative_counterfactual_user(item) ==
        "The coin was kept in the drawer. UPDATE: A courier transfers it to the drawer. Where is the coin now?");
    item.expected = "The basket.";
    rejected = false;
    try { (void) recirculation::alternative_counterfactual_user(item); }
    catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    const std::vector<std::vector<float>> samples{{1, 0}, {0, 1}, {1, 0}};
    const auto leave_one_out = recirculation::alternative_mean_direction(samples, 1);
    assert(leave_one_out.size() == 2 && std::abs(leave_one_out[0] - 1.0f) < 1e-6f);
    assert(std::abs(*recirculation::alternative_alignment(samples[1], leave_one_out)) < 1e-6);
    assert(!recirculation::alternative_alignment({0, 0}, leave_one_out));
}
