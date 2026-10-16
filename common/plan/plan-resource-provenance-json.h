#pragma once

#include "resource/resource-contract.h"
#include <nlohmann/json.hpp>

// Additive fields shared by persistent plans. Older records read as unknown.
inline void common_plan_resource_provenance_to_json(const common_runtime_resource_ref & resource,
        nlohmann::ordered_json & value) {
    auto & metadata=value["metadata"];
    metadata["source_provider"]=resource.metadata.source_provider;
    metadata["source_operation"]=resource.metadata.source_operation;
    metadata["source_request_json"]=resource.metadata.source_request_json;
    metadata["retrieved_at"]=resource.metadata.retrieved_at;
    metadata["content_hash"]=resource.metadata.content_hash;
    value["lineage"]={{"parent_uri",resource.lineage.parent_uri},
        {"chunk_index",resource.lineage.chunk_index},{"chunk_count",resource.lineage.chunk_count},
        {"byte_offset",resource.lineage.byte_offset},{"byte_length",resource.lineage.byte_length},
        {"overlap_bytes",resource.lineage.overlap_bytes},{"derivation",resource.lineage.derivation}};
}

inline void common_plan_resource_provenance_from_json(const nlohmann::ordered_json & value,
        common_runtime_resource_ref & resource) {
    if(value.contains("metadata")&&value["metadata"].is_object()){
        const auto & metadata=value["metadata"];
        resource.metadata.source_provider=metadata.value("source_provider",std::string{});
        resource.metadata.source_operation=metadata.value("source_operation",std::string{});
        resource.metadata.source_request_json=metadata.value("source_request_json",std::string{});
        resource.metadata.retrieved_at=metadata.value("retrieved_at",int64_t(0));
        resource.metadata.content_hash=metadata.value("content_hash",std::string{});
    }
    if(value.contains("lineage")&&value["lineage"].is_object()){
        const auto & lineage=value["lineage"];
        resource.lineage.parent_uri=lineage.value("parent_uri",std::string{});
        resource.lineage.chunk_index=lineage.value("chunk_index",size_t(0));
        resource.lineage.chunk_count=lineage.value("chunk_count",size_t(0));
        resource.lineage.byte_offset=lineage.value("byte_offset",size_t(0));
        resource.lineage.byte_length=lineage.value("byte_length",size_t(0));
        resource.lineage.overlap_bytes=lineage.value("overlap_bytes",size_t(0));
        resource.lineage.derivation=lineage.value("derivation",std::string{});
    }
}
