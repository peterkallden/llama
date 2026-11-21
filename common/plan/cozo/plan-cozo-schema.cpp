#include "plan/cozo/plan-cozo-schema.h"
std::string common_plan_cozo_schema_script() { return R"COZO(
{ ?[id, state_json] <- [['__schema_probe__', '{}']] :create agent_plan { id: String => state_json: String } }
{ ?[plan_id, sequence, event_json] <- [['__schema_probe__', 0, '{}']] :create agent_plan_event { plan_id: String, sequence: Int => event_json: String } }
{ ?[id] <- [['__schema_probe__']] :delete agent_plan { id } }
{ ?[plan_id, sequence] <- [['__schema_probe__', 0]] :delete agent_plan_event { plan_id, sequence } }
)COZO"; }
