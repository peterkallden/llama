#include "agent/adaptation/flydelta/flydelta-two-pass.h"

#include <deque>

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

namespace {

class fake_inference final : public common_agent_inference {
public:
    std::vector<common_agent_generation_request> seen;

    bool generate(
            const common_agent_generation_request & request,
            common_agent_generation_result & result) override {
        seen.push_back(request);
        result = {};
        result.status = common_agent_generation_status::completed;
        result.stop_reason = common_agent_generation_stop_reason::eos;
        result.content = seen.size() == 1 ? "capture" : "final";
        if (seen.size() == 1) {
            auto capture = std::make_shared<common_flydelta_hidden_state_capture>();
            capture->captured = true;
            capture->model_profile_fingerprint = "model:test";
            capture->capture_layout_revision = "layer-input:v1";
            capture->layer_indices = {1};
            capture->n_embd = 2;
            capture->token_index = 0;
            capture->values = {0.25f, -0.5f};
            result.flydelta_capture = std::move(capture);
        }
        return true;
    }
};

} // namespace

int main() {
    auto capture = std::make_shared<common_flydelta_hidden_state_capture_request>();
    capture->enabled = true;
    capture->layer_indices = {1};
    capture->max_bytes = 4096;
    capture->model_profile_fingerprint = "model:test";
    capture->capture_layout_revision = "layer-input:v1";

    common_agent_generation_request request;
    request.purpose = common_agent_generation_purpose::conversation;
    request.options.n_predict = 4;
    request.messages = {{"user", "test"}};

    fake_inference inference;
    common_flydelta_two_pass_result result;
    std::string error;
    CHECK(common_flydelta_run_two_pass(
        inference, request, capture,
        [](const common_flydelta_hidden_state_capture & value,
                common_flydelta_activation_result & activation,
                std::string & builder_error) {
            if (value.values.size() != 2) {
                builder_error = "unexpected capture size";
                return false;
            }
            activation = {};
            return true;
        }, result, error));
    CHECK(inference.seen.size() == 2);
    CHECK(inference.seen[0].flydelta_capture == capture);
    CHECK(!inference.seen[0].flydelta_activation);
    CHECK(!inference.seen[1].flydelta_capture);
    CHECK(inference.seen[1].flydelta_activation != nullptr);
    CHECK(!request.flydelta_capture);
    CHECK(!request.flydelta_activation);
    CHECK(result.capture_generation.content == "capture");
    CHECK(result.final_generation.content == "final");
    return 0;
}
