#include "memory/cozo/cozo-schema.h"

std::string common_memory_cozo_schema_script() {
    return R"COZO(
        {
            ?[id, kind, content, summary, embedding, importance, confidence, created_at, accessed_at, access_count, metadata_json] <-
                [['__schema_probe__', 'fact', '', '', [], 0.0, 0.0, 0, 0, 0, '{}']]
            :create memory {
                id: String =>
                kind: String,
                content: String,
                summary: String,
                embedding: <F32; 0>?,
                importance: Float,
                confidence: Float,
                created_at: Int,
                accessed_at: Int,
                access_count: Int,
                metadata_json: String
            }
        }
        {
            ?[from, relation, to, weight, created_at] <-
                [['__schema_probe__', 'related', '__schema_probe__', 0.0, 0]]
            :create memory_edge {
                from: String,
                relation: String,
                to: String =>
                weight: Float,
                created_at: Int
            }
        }
        {
            ?[id] <- [['__schema_probe__']]
            :delete memory { id }
        }
        {
            ?[from, relation, to] <- [['__schema_probe__', 'related', '__schema_probe__']]
            :delete memory_edge { from, relation, to }
        }
    )COZO";
}
