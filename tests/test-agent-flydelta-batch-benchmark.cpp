#include "agent/adaptation/flydelta/flydelta-model-adapter.h"

#include <chrono>
#include <cstdlib>
#include <iostream>

namespace {

common_flydelta_arm_result execute_reference_arm(
        const common_flydelta_arm_request & request) {
    common_flydelta_arm_result result;
    result.arm_id = request.arm_id;
    result.executed = true;
    result.requested_alpha = request.alpha;
    result.executed_alpha = request.alpha;
    result.execution_metrics.available = true;
    result.execution_metrics.model_ms = 0.01f;
    result.execution_metrics.diagnostics_bytes_to_host = sizeof(float) * 4;
    return result;
}

common_flydelta_arm_batch_request make_request(size_t count) {
    common_flydelta_arm_batch_request request;
    request.arms.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        common_flydelta_arm_request arm;
        arm.arm_id = "flydelta://benchmark/arm/" + std::to_string(index);
        arm.alpha = 0.01f + static_cast<float>(index) * 0.001f;
        request.arms.push_back(std::move(arm));
    }
    return request;
}

template <typename Function>
long long measure_us(Function && function, size_t repetitions) {
    const auto start = std::chrono::steady_clock::now();
    for (size_t index = 0; index < repetitions; ++index) function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

} // namespace

int main() {
    constexpr size_t arm_count = 32;
    constexpr size_t repetitions = 2000;
    const auto request = make_request(arm_count);
    std::string error;

    common_flydelta_model_host scalar_host;
    scalar_host.run_bounded_arm = [](const auto & arm, auto & result, auto &) {
        result = execute_reference_arm(arm);
        return true;
    };

    common_flydelta_model_host batch_host = scalar_host;
    batch_host.run_bounded_arm_batch = [](const auto & batch, auto & result, auto &) {
        result = {};
        result.arms.reserve(batch.arms.size());
        for (const auto & arm : batch.arms) {
            result.arms.push_back(execute_reference_arm(arm));
        }
        return true;
    };

    common_flydelta_arm_batch_result result;
    if (!common_flydelta_run_bounded_arm_batch(scalar_host, request, result, error) ||
            !common_flydelta_run_bounded_arm_batch(batch_host, request, result, error)) {
        std::cerr << "FlyDelta batch benchmark setup failed: " << error << '\n';
        return 1;
    }

    const long long scalar_us = measure_us([&]() {
        common_flydelta_arm_batch_result ignored;
        std::string ignored_error;
        if (!common_flydelta_run_bounded_arm_batch(
                scalar_host, request, ignored, ignored_error)) std::abort();
    }, repetitions);
    const long long batch_us = measure_us([&]() {
        common_flydelta_arm_batch_result ignored;
        std::string ignored_error;
        if (!common_flydelta_run_bounded_arm_batch(
                batch_host, request, ignored, ignored_error)) std::abort();
    }, repetitions);

    std::cout << "flydelta_batch_benchmark kind=callback-seam"
              << " arms=" << arm_count
              << " repetitions=" << repetitions
              << " scalar_us=" << scalar_us
              << " batch_us=" << batch_us
              << " note=not-device-throughput"
              << " scalar_path=scalar_fallback"
              << " batch_path=backend_batch" << '\n';
    return 0;
}
