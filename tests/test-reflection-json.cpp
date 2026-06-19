#include "agent/reflection-json.h"
#include <cassert>
int main() { common_reflection_result result; std::string error; assert(common_reflection_parse_json(R"({"decision":"revise","ready_to_answer":false,"revision_guidance":["cite evidence"]})", result, error)); assert(result.decision == common_reflection_decision::revise); assert(!common_reflection_parse_json("not json", result, error)); assert(!common_reflection_parse_json(R"({"decision":"unsafe"})", result, error)); return 0; }
