#include "historical-window.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

int main() {
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
}
