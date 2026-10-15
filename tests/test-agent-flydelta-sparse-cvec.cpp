#include "server-task.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

std::shared_ptr<server_task_cvec> make_cvec(const char * identity, uint32_t layer_offset) {
    auto value = std::make_shared<server_task_cvec>();
    value->identity = identity;
    value->content_hash = std::string("sha256:") + identity;
    value->n_embd = 4;
    value->il_start = 1;
    value->il_end = 5;
    value->data.assign(static_cast<size_t>(value->n_embd) * 5, 0.0f);
    value->sparse_layer_indices = {2, 4};
    value->sparse_data.resize(8);
    for (size_t index = 0; index < value->sparse_data.size(); ++index) {
        value->sparse_data[index] = static_cast<float>(layer_offset + index + 1);
    }
    return value;
}

} // namespace

int main() {
    constexpr size_t max_bytes = 1024 * 1024;
    std::string error;
    auto first = make_cvec("sparse-a", 0);
    auto second = make_cvec("sparse-b", 100);

    if (!server_task_cvec_validate(*first, 4, 6, max_bytes, error) ||
            !server_task_cvec_validate(*second, 4, 6, max_bytes, error)) {
        std::cerr << "valid sparse cvec rejected: " << error << '\n';
        return 1;
    }

    server_task_cvec_batch batch;
    if (!batch.add(10, first, error) || !batch.add(11, second, error)) {
        std::cerr << "compatible sparse rows rejected: " << error << '\n';
        return 1;
    }
    if (!batch.has_sparse() || batch.size() != 2) {
        std::cerr << "sparse batch metadata was not retained\n";
        return 1;
    }

    std::vector<llama_seq_id> seq_ids;
    std::vector<const float *> data;
    std::vector<uint32_t> layers;
    size_t data_len = 0;
    int32_t n_embd = 0;
    if (!batch.materialize_sparse(seq_ids, data, data_len, n_embd, layers, error) ||
            seq_ids != std::vector<llama_seq_id>{10, 11} ||
            data.size() != 2 || data_len != 8 || n_embd != 4 ||
            layers != std::vector<uint32_t>{2, 4} ||
            data[0][0] == data[1][0]) {
        std::cerr << "sparse batch materialization failed: " << error << '\n';
        return 1;
    }

    // A one-row sparse request is a valid device-table input too. The server
    // uses this to avoid expanding an otherwise sparse scalar overlay into
    // an all-layer cvec solely because the current ubatch has one sequence.
    server_task_cvec_batch single;
    if (!single.add(10, first, error) || !single.has_sparse() || single.size() != 1 ||
            !single.materialize_sparse(seq_ids, data, data_len, n_embd, layers, error) ||
            seq_ids != std::vector<llama_seq_id>{10} || data.size() != 1 ||
            data_len != 8 || n_embd != 4 || layers != std::vector<uint32_t>{2, 4}) {
        std::cerr << "single sparse row materialization failed: " << error << '\n';
        return 1;
    }

    // A factorized wave shares B and transports only one c-row per sequence.
    auto basis = std::make_shared<server_task_cvec::factorized_basis_data>();
    basis->rank = 2;
    basis->layer_indices = {4};
    basis->fingerprint = "fixture-basis-fingerprint";
    basis->data = std::make_shared<const std::vector<float>>(
        std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    auto factor_a = make_cvec("factor-a", 0);
    auto factor_b = make_cvec("factor-b", 0);
    factor_a->factorized_basis = basis;
    factor_b->factorized_basis = basis;
    factor_a->factorized_coefficients = {0.25f, 0.0f};
    factor_b->factorized_coefficients = {0.0f, -0.5f};
    for (auto * row : {factor_a.get(), factor_b.get()}) {
        row->execution_batch_id = "batch-1";
        row->execution_wave_id = "wave-1";
        row->execution_phase = "deep_controls";
        row->execution_strategy = "coordinate";
    }
    server_task_cvec_batch factorized;
    std::shared_ptr<const server_task_cvec::factorized_basis_data> material_basis;
    std::vector<const float *> coefficient_rows;
    if (!factorized.add(10, factor_a, error) || !factorized.add(11, factor_b, error) ||
            !factorized.has_factorized() ||
            !factorized.materialize_factorized(seq_ids, material_basis, coefficient_rows, n_embd, error) ||
            material_basis != basis || coefficient_rows.size() != 2 || n_embd != 4 ||
            coefficient_rows[0][0] != 0.25f || coefficient_rows[1][1] != -0.5f) {
        std::cerr << "factorized batch materialization failed: " << error << '\n';
        return 1;
    }

    auto wrong_wave = make_cvec("factor-wrong-wave", 0);
    wrong_wave->factorized_basis = basis;
    wrong_wave->factorized_coefficients = factor_a->factorized_coefficients;
    wrong_wave->execution_batch_id = factor_a->execution_batch_id;
    wrong_wave->execution_wave_id = "wave-2";
    wrong_wave->execution_phase = factor_a->execution_phase;
    wrong_wave->execution_strategy = factor_a->execution_strategy;
    server_task_cvec_batch mixed_wave;
    if (!mixed_wave.add(10, factor_a, error) || mixed_wave.add(16, wrong_wave, error)) {
        std::cerr << "different coefficient waves were accepted in one physical batch\n";
        return 1;
    }
    auto mismatched_fingerprint_basis = std::make_shared<server_task_cvec::factorized_basis_data>(*basis);
    mismatched_fingerprint_basis->fingerprint = "other-basis-fingerprint";
    auto mismatched_fingerprint = make_cvec("factor-mismatched-fingerprint", 0);
    mismatched_fingerprint->factorized_basis = mismatched_fingerprint_basis;
    mismatched_fingerprint->factorized_coefficients = factor_a->factorized_coefficients;
    mismatched_fingerprint->execution_batch_id = factor_a->execution_batch_id;
    mismatched_fingerprint->execution_wave_id = factor_a->execution_wave_id;
    mismatched_fingerprint->execution_phase = factor_a->execution_phase;
    mismatched_fingerprint->execution_strategy = factor_a->execution_strategy;
    server_task_cvec_batch mismatched_basis;
    if (!mismatched_basis.add(10, factor_a, error) ||
            !mismatched_basis.add(17, mismatched_fingerprint, error) ||
            mismatched_basis.materialize_factorized(
                seq_ids, material_basis, coefficient_rows, n_embd, error)) {
        std::cerr << "different basis fingerprints were accepted in one factorized wave\n";
        return 1;
    }
    std::vector<std::vector<float>> owned_rows;
    std::vector<const float *> fallback_rows;
    if (!factorized.materialize_factorized_sparse(
                seq_ids, owned_rows, fallback_rows, n_embd, layers, error) ||
            seq_ids != std::vector<llama_seq_id>{10, 11} ||
            layers != std::vector<uint32_t>{4} || fallback_rows.size() != 2 ||
            owned_rows.size() != 2 || owned_rows[0].size() != 4 ||
            std::fabs(fallback_rows[0][0] - 0.25f) > 1.0e-7f ||
            std::fabs(fallback_rows[1][1] + 0.5f) > 1.0e-7f) {
        std::cerr << "factorized sparse fallback materialization failed: " << error << '\n';
        return 1;
    }

    // Native binding rejects a non-shared basis, but the fallback still
    // materializes every row over the union of the layers.
    auto other_basis = std::make_shared<server_task_cvec::factorized_basis_data>();
    other_basis->rank = 1;
    other_basis->layer_indices = {2};
    other_basis->data = std::make_shared<const std::vector<float>>(
        std::vector<float>{0.0f, 0.0f, 0.0f, 2.0f});
    auto heterogeneous = make_cvec("factor-c", 0);
    heterogeneous->factorized_basis = other_basis;
    heterogeneous->factorized_coefficients = {0.5f};
    heterogeneous->execution_batch_id = factor_a->execution_batch_id;
    heterogeneous->execution_wave_id = factor_a->execution_wave_id;
    heterogeneous->execution_phase = factor_a->execution_phase;
    heterogeneous->execution_strategy = factor_a->execution_strategy;
    server_task_cvec_batch mixed_basis;
    if (!mixed_basis.add(10, factor_a, error) ||
            !mixed_basis.add(12, heterogeneous, error) ||
            mixed_basis.materialize_factorized(
                seq_ids, material_basis, coefficient_rows, n_embd, error) ||
            !mixed_basis.materialize_factorized_sparse(
                seq_ids, owned_rows, fallback_rows, n_embd, layers, error) ||
            layers != std::vector<uint32_t>{2, 4} || owned_rows.size() != 2 ||
            std::fabs(owned_rows[1][3] - 1.0f) > 1.0e-7f ||
            std::fabs(owned_rows[0][4] - 0.25f) > 1.0e-7f) {
        std::cerr << "heterogeneous factorized sparse fallback failed: " << error << '\n';
        return 1;
    }

    // Singleton tails and identical coefficient rows remain valid; neither
    // case is permitted to degrade into an empty scalar vector.
    server_task_cvec_batch singleton_factorized;
    if (!singleton_factorized.add(13, factor_a, error) ||
            !singleton_factorized.materialize_factorized_sparse(
                seq_ids, owned_rows, fallback_rows, n_embd, layers, error) ||
            seq_ids != std::vector<llama_seq_id>{13} || owned_rows.size() != 1 ||
            owned_rows[0].size() != 4 || std::fabs(owned_rows[0][0] - 0.25f) > 1.0e-7f) {
        std::cerr << "singleton factorized tail materialization failed: " << error << '\n';
        return 1;
    }
    auto factor_identical = make_cvec("factor-identical", 0);
    factor_identical->factorized_basis = basis;
    factor_identical->factorized_coefficients = factor_a->factorized_coefficients;
    factor_identical->execution_batch_id = factor_a->execution_batch_id;
    factor_identical->execution_wave_id = factor_a->execution_wave_id;
    factor_identical->execution_phase = factor_a->execution_phase;
    factor_identical->execution_strategy = factor_a->execution_strategy;
    server_task_cvec_batch identical;
    if (!identical.add(14, factor_a, error) || !identical.add(15, factor_identical, error) ||
            !identical.materialize_factorized_sparse(
                seq_ids, owned_rows, fallback_rows, n_embd, layers, error) ||
            owned_rows.size() != 2 || owned_rows[0] != owned_rows[1]) {
        std::cerr << "identical factorized rows were not preserved: " << error << '\n';
        return 1;
    }

    auto incompatible = make_cvec("sparse-c", 200);
    incompatible->sparse_layer_indices = {3, 4};
    server_task_cvec_batch rejected;
    if (!rejected.add(10, first, error) || rejected.add(11, incompatible, error)) {
        std::cerr << "incompatible sparse layer layout was accepted\n";
        return 1;
    }

    std::cout << "agent_flydelta_sparse_cvec=passed\n";
    return 0;
}
