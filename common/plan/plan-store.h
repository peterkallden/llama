#pragma once
#include "plan/plan-types.h"
#include <optional>
#include <utility>
class common_plan_store { public: virtual ~common_plan_store() = default; virtual bool open(const std::string & path, std::string & error) = 0; virtual void close() = 0; virtual bool create(const common_plan_state & plan, std::string & error) = 0; virtual std::optional<common_plan_state> get(const std::string & plan_id, std::string & error) = 0; virtual std::vector<common_plan_state> list(std::string & error) = 0; virtual bool apply(const common_plan_operation & operation, common_plan_state & updated_plan, std::string & error) = 0; virtual std::vector<common_plan_event> history(const std::string & plan_id, std::string & error) = 0; virtual bool erase(const std::string & plan_id, std::string & error) = 0; };

// Host-owned in-place route migration. This is intentionally a store-level
// helper so reflection/replan code cannot accidentally turn a route change
// into an ordinary model proposal.
inline bool common_plan_request_route_transition(
        common_plan_store & store,
        const std::string & plan_id,
        const common_plan_route_binding & binding,
        const common_plan_workflow_definition & workflow_definition,
        const std::vector<std::string> & required_capabilities,
        const std::vector<std::string> & procedure_refs,
        const std::string & reason,
        common_plan_state & updated_plan,
        std::string & error,
        std::optional<common_plan_workflow_continuation_record> continuation = std::nullopt) {
    auto current = store.get(plan_id, error);
    if (!current) return false;
    common_plan_operation operation;
    operation.kind = common_plan_operation_kind::request_route_transition;
    operation.plan_id = plan_id;
    operation.expected_version = current->version;
    operation.reason_summary = reason;
    operation.route_binding = binding;
    operation.workflow_definition = workflow_definition;
    operation.route_required_capabilities = required_capabilities;
    operation.route_procedure_refs = procedure_refs;
    operation.workflow_continuation_record = std::move(continuation);
    operation.host_authorized = true;
    return store.apply(operation, updated_plan, error);
}
