#include "plan/plan-memory.h"

#include <algorithm>
#include <set>

void common_plan_bind_memory_provenance(
        common_plan_operation & operation,
        const std::vector<common_memory_hit> & memories) {
    if (operation.kind != common_plan_operation_kind::add_step || !operation.step) return;
    std::set<std::string> eligible;
    for (const auto & hit : memories) {
        if (hit.memory.kind == common_memory_kind::procedure) eligible.insert(hit.memory.id);
    }
    auto & ids = operation.step->source_memory_ids;
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](const std::string & id) { return !eligible.count(id); }), ids.end());
    operation.step->generated_from_memory = !ids.empty();
    for (const auto & id : ids) {
        const auto evidence = "memory:" + id;
        if (std::find(operation.evidence_ids.begin(), operation.evidence_ids.end(), evidence) == operation.evidence_ids.end()) {
            operation.evidence_ids.push_back(evidence);
        }
    }
}
