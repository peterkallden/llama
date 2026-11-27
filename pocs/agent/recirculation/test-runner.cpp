#include "runner.h"
#include <cassert>

struct fixture final : recirculation::backend {
    int fault = -1, discarded = 0, cleared = 0, last = -1;
    bool hooked = false;
    bool omit_destination = false;
    std::vector<int> calls;
    std::vector<int32_t> captured_positions;
    std::vector<std::vector<float>> destinations;
    bool ready(int32_t pos) override { calls.push_back(0); return fault != 0 && last == pos - 1; }
    bool first(int32_t, int32_t pos, bool capture_inputs, recirculation::capture & out) override {
        calls.push_back(1); if (capture_inputs) captured_positions.push_back(pos); hooked = true; last = pos;
        out = {{1.0f, 2.0f}, {3.0f, 4.0f}, {0.0f, 2.0f}};
        if (size_t(pos) < destinations.size()) out.destination = destinations[size_t(pos)];
        if (omit_destination && capture_inputs) out.destination.clear();
        if (fault == 2) out.source.clear();
        if (fault == 6) throw std::runtime_error("abort");
        return fault != 1;
    }
    bool rewind(int32_t pos) override { calls.push_back(3); --last; return fault != 3 && last == pos - 1; }
    bool second(int32_t, int32_t pos, const std::vector<float> & patch, bool & matched) override {
        calls.push_back(4); last = pos;
        assert(patch.size() == 2);
        matched = fault != 5;
        return fault != 4;
    }
    void clear_hooks() noexcept override { hooked = false; ++cleared; }
    void discard() noexcept override { ++discarded; }
};
int main() {
    const recirculation_config config{"fixture", 2, 1, 0.1f, 1.0f, 1};
    for (int fault = -1; fault <= 6; ++fault) {
        fixture f; f.fault = fault;
        recirculation::runner run(f, config, 4, 2);
        std::vector<float> logits;
        recirculation::observation obs;
        const bool ok = run.step(7, 0, logits, obs);
        assert(ok == (fault == -1));
        assert(!f.hooked && f.cleared > 0);
        if (ok) {
            assert((f.calls == std::vector<int>{0, 1, 3, 4}));
            assert((logits == std::vector<float>{1, 2}));
            assert(obs.first_completed && obs.second_completed && obs.patch_matched && obs.committed);
            assert(std::abs(obs.source_destination_cosine - 0.8f) < 1e-6f);
            assert(std::abs(obs.delta_norm - 0.5f) < 1e-6f);
            for (int i = 1; i < 128; ++i) assert(run.step(7, i, logits, obs));
            assert(f.last == 127 && f.discarded == 0);
        } else {
            assert(f.discarded == 1 && logits.empty() && !obs.committed && !obs.failure.empty());
            const auto calls = f.calls;
            assert(!run.step(7, 0, logits, obs));
            assert(f.calls == calls);
        }
    }
    fixture disabled;
    recirculation::runner plain(disabled, config, 4, 2, false);
    recirculation::observation obs;
    std::vector<float> logits;
    assert(plain.step(7, 0, logits, obs));
    assert((disabled.calls == std::vector<int>{0, 1}));
    assert(obs.committed && !obs.second_completed);
    // Configuration is copied at construction, not read through a mutable reference.
    fixture owned;
    auto mutable_config = config;
    recirculation::runner snapshot(owned, mutable_config, 4, 2);
    mutable_config.source_layer_input = 100;
    assert(snapshot.step(7, 0, logits, obs));
    fixture isolated;
    recirculation::runner other(isolated, config, 4, 2);
    assert(!other.step(7, 1, logits, obs));
    assert(isolated.calls.empty() && isolated.discarded == 1);
    fixture targeted;
    recirculation::runner selected(targeted, config, 4, 2, true,
        recirculation::patch_policy::selected_position, 1);
    assert(selected.step(7, 0, logits, obs) && !obs.second_completed);
    assert(selected.step(7, 1, logits, obs) && obs.second_completed);
    assert((targeted.calls == std::vector<int>{0, 1, 0, 1, 3, 4}));
    assert((targeted.captured_positions == std::vector<int32_t>{1}));
    fixture controlled;
    recirculation::runner control(controlled, config, 4, 2, true,
        recirculation::patch_policy::selected_position, 0, {5.0f, 12.0f});
    assert(control.step(7, 0, logits, obs) && obs.second_completed && obs.source_norm == 2.0f);
    bool rejected = false;
    try {
        recirculation::runner invalid(targeted, config, 4, 2, true,
            recirculation::patch_policy::selected_position, -1);
    } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    fixture window;
    recirculation::target_policy window_target{recirculation::patch_policy::host_window, 1, 2, 0.15f};
    recirculation::runner window_run(window, config, 4, 2, true, window_target.mode, -1, {}, window_target);
    for (int i = 0; i < 4; ++i) {
        assert(window_run.step(7, i, logits, obs));
        assert(obs.second_completed == (i == 2));
    }
    assert((window.captured_positions == std::vector<int32_t>{2}));
    fixture novelty;
    novelty.destinations = {{1, 0}, {1, 0}, {0, 1}, {1, 0}};
    recirculation::target_policy novelty_target{recirculation::patch_policy::activation_novelty, 0, 3, 1.0f};
    recirculation::runner novelty_run(novelty, config, 4, 2, true, novelty_target.mode, -1, {}, novelty_target);
    for (int i = 0; i < 4; ++i) {
        assert(novelty_run.step(7, i, logits, obs));
        assert(obs.second_completed == (i == 2));
    }
    fixture zero;
    zero.destinations = {{0, 0}, {1, 0}};
    recirculation::runner no_trigger(zero, config, 4, 2, true, novelty_target.mode, -1, {},
        {novelty_target.mode, 0, 1, 0.1f});
    assert(no_trigger.step(7, 0, logits, obs) && !obs.second_completed);
    assert(no_trigger.step(7, 1, logits, obs) && !obs.second_completed);
    fixture boundary;
    boundary.destinations = {{1, 0}, {0, 1}};
    recirculation::runner boundary_run(boundary, config, 4, 2, true, novelty_target.mode, -1, {},
        {novelty_target.mode, 0, 1, 1.0f});
    assert(boundary_run.step(7, 0, logits, obs) && !obs.second_completed);
    assert(boundary_run.step(7, 1, logits, obs) && obs.second_completed && obs.novelty == 1.0f);
    fixture missing;
    missing.omit_destination = true;
    recirculation::runner missing_run(missing, config, 4, 2, true, novelty_target.mode, -1, {}, novelty_target);
    assert(!missing_run.step(7, 0, logits, obs) && obs.failure == "missing_capture" && missing.discarded == 1);
    rejected = false;
    try {
        recirculation::runner invalid_window(missing, config, 4, 2, true, novelty_target.mode, -1, {},
            {novelty_target.mode, 2, 1, 0.15f});
    } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
    rejected = false;
    try {
        recirculation::runner invalid(targeted, config, 4, 2, true,
            recirculation::patch_policy::selected_position, 1, {1.0f});
    } catch (const std::invalid_argument &) { rejected = true; }
    assert(rejected);
}
