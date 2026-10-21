#pragma once

// Internal Cozo relation definitions. They are deliberately kept outside the
// public data-store contract so another backend does not inherit Cozo layout.
const char * agent_cozo_schema_script();
const char * agent_cozo_values_schema_script();
const char * agent_cozo_order_schema_script();
const char * agent_cozo_dataset_schema_script();
const char * agent_cozo_investigation_schema_script();
const char * agent_cozo_investigation_scope_index_schema_script();
const char * agent_cozo_investigation_report_index_schema_script();
