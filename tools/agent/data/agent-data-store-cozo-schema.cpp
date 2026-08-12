#include "agent-data-store-cozo-schema.h"

namespace {

const char * schema = R"COZO(
    {
        ?[dataset, row_id, row_json] <- [['__probe__', '__probe__', '{}']]
        :create agent_data_rows {
            dataset: String,
            row_id: String =>
            row_json: String
        }
    }
    {
        ?[dataset, row_id] <- [['__probe__', '__probe__']]
        :delete agent_data_rows { dataset, row_id }
    }
    {
        ?[dataset, row_id, field, value_kind, value_text, value_number] <- [['__probe__', '__probe__', '__probe__', 'string', '', 0.0]]
        :create agent_data_values {
            dataset: String,
            row_id: String,
            field: String =>
            value_kind: String,
            value_text: String,
            value_number: Float
        }
    }
    {
        ?[dataset, row_id, field] <- [['__probe__', '__probe__', '__probe__']]
        :delete agent_data_values { dataset, row_id, field }
    }
    {
        ?[dataset, row_id, row_seq] <- [['__probe__', '__probe__', 1]]
        :create agent_data_row_order {
            dataset: String,
            row_id: String =>
            row_seq: Int
        }
    }
    {
        ?[dataset, row_id] <- [['__probe__', '__probe__']]
        :delete agent_data_row_order { dataset, row_id }
    }
)COZO";

const char * values_schema = R"COZO(
    {
        ?[dataset, row_id, field, value_kind, value_text, value_number] <- [['__probe__', '__probe__', '__probe__', 'string', '', 0.0]]
        :create agent_data_values {
            dataset: String,
            row_id: String,
            field: String =>
            value_kind: String,
            value_text: String,
            value_number: Float
        }
    }
    {
        ?[dataset, row_id, field] <- [['__probe__', '__probe__', '__probe__']]
        :delete agent_data_values { dataset, row_id, field }
    }
)COZO";

const char * order_schema = R"COZO(
    {
        ?[dataset, row_id, row_seq] <- [['__probe__', '__probe__', 1]]
        :create agent_data_row_order {
            dataset: String,
            row_id: String =>
            row_seq: Int
        }
    }
    {
        ?[dataset, row_id] <- [['__probe__', '__probe__']]
        :delete agent_data_row_order { dataset, row_id }
    }
)COZO";

const char * dataset_schema = R"COZO(
    {
        ?[dataset_uri, descriptor_json] <- [['__probe__', '{}']]
        :create agent_dataset_metadata {
            dataset_uri: String =>
            descriptor_json: String
        }
    }
    {
        ?[dataset_uri] <- [['__probe__']]
        :delete agent_dataset_metadata { dataset_uri }
    }
)COZO";

} // namespace

const char * agent_cozo_schema_script() { return schema; }
const char * agent_cozo_values_schema_script() { return values_schema; }
const char * agent_cozo_order_schema_script() { return order_schema; }
const char * agent_cozo_dataset_schema_script() { return dataset_schema; }
