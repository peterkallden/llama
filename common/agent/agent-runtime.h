#pragma once
#include "memory/memory-types.h"
#include "agent/tool-registry.h"
#include "plan/plan-store.h"
#include <memory>

enum class common_reflection_decision { accept, revise, request_action, replan, abort };
struct common_reflection_issue { std::string kind, description, correction; float severity = 0.5f; };
struct common_reflection_result { common_reflection_decision decision = common_reflection_decision::accept; std::vector<common_reflection_issue> issues; std::vector<common_plan_operation> proposed_plan_operations; std::optional<std::string> requested_action; std::vector<std::string> revision_guidance; bool ready_to_answer = false; float confidence = 0.5f; };
struct common_plan_proposal { common_plan_state plan; std::vector<common_plan_operation> operations; };
struct common_agent_request { std::string prompt, session_id; std::vector<common_memory_hit> memories; std::optional<common_registered_tool_call> tool_call; size_t max_iterations = 2, max_reflection_rounds = 1, max_tool_batches = 1; };
struct common_agent_result { std::string answer, error, plan_id; uint64_t plan_version = 0; bool revised = false, limit_reached = false; };
class common_planner { public: virtual ~common_planner() = default; virtual common_plan_proposal create_plan(const common_agent_request & request, std::string & error) = 0; };
class common_action_executor { public: virtual ~common_action_executor() = default; virtual std::string generate_draft(const common_agent_request & request, const common_plan_state & plan, const std::vector<std::string> & guidance, std::string & error) = 0; };
class common_reflection_engine { public: virtual ~common_reflection_engine() = default; virtual common_reflection_result evaluate(const common_agent_request & request, const common_plan_state & plan, const std::string & draft, std::string & error) = 0; };
class common_agent_runtime { public: common_agent_runtime(common_plan_store & store, common_planner & planner, common_action_executor & executor, common_reflection_engine & reflector, const common_tool_registry * tools = nullptr); common_agent_result run(const common_agent_request & request); private: common_plan_store & store; common_planner & planner; common_action_executor & executor; common_reflection_engine & reflector; const common_tool_registry * tools; };
