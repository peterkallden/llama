#include "agent/adaptation/flydelta/flydelta-gate.h"

#define CHECK(condition) do { if (!(condition)) return __LINE__; } while (false)

int main() {
    std::string error;
    common_flydelta_gate_config config;
    config.enabled = true;
    common_flydelta_gate_request request;
    request.explicit_opt_in = true;
    request.candidate_status = common_flydelta_candidate_status::approved;
    request.basis_available = true;
    request.familiarity = 0.95f;
    request.novelty = 0.05f;
    request.requested_scale = 0.10f;
    common_flydelta_gate_decision decision;
    CHECK(common_flydelta_gate_decide(config, request, decision, error));
    CHECK(decision.apply && decision.scale == 0.10f);

    request.explicit_opt_in = false;
    CHECK(common_flydelta_gate_decide(config, request, decision, error));
    CHECK(!decision.apply && decision.reason == "explicit_opt_in_required");
    request.explicit_opt_in = true;
    request.candidate_status = common_flydelta_candidate_status::eligible;
    CHECK(common_flydelta_gate_decide(config, request, decision, error));
    CHECK(!decision.apply && decision.reason == "candidate_not_approved");
    request.candidate_status = common_flydelta_candidate_status::approved;
    request.familiarity = 0.10f;
    request.novelty = 0.90f;
    CHECK(common_flydelta_gate_decide(config, request, decision, error));
    CHECK(!decision.apply && decision.reason == "context_not_familiar");

    request.familiarity = 0.95f;
    request.novelty = 0.05f;
    request.requested_scale = 0.0f;
    CHECK(common_flydelta_gate_decide(config, request, decision, error));
    CHECK(!decision.apply && decision.reason == "requested_scale_zero");
    config.max_novelty = 0.0f;
    config.min_familiarity = 0.0f;
    CHECK(!common_flydelta_gate_config_validate(config, error));
    return 0;
}
