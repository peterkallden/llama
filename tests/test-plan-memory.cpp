#include "plan/plan-memory.h"

#include <cassert>

int main() {
    common_plan_operation operation;
    operation.kind = common_plan_operation_kind::add_step;
    operation.step = common_plan_step{"verify", "Verify", "Verify persistence"};
    operation.step->source_memory_ids = {"procedure-1", "invented"};
    common_memory_hit procedure;
    procedure.memory.id = "procedure-1";
    procedure.memory.kind = common_memory_kind::procedure;
    common_memory_hit fact;
    fact.memory.id = "fact-1";
    fact.memory.kind = common_memory_kind::fact;
    common_plan_bind_memory_provenance(operation, {procedure, fact});
    assert(operation.step->generated_from_memory);
    assert(operation.step->source_memory_ids == std::vector<std::string>{"procedure-1"});
    assert(operation.evidence_ids == std::vector<std::string>{"memory:procedure-1"});
    return 0;
}
