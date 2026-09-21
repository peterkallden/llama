#include "server-task.h"

#include <algorithm>
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
