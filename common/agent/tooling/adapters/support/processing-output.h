#pragma once

#include "agent/tooling/adapters/tool-adapters.h"

// The processing service persists outputs and returns resource descriptors;
// direct providers may return raw output. Read either through the same bounds.
inline bool common_adapter_read_processing_output(
        const agent_resource_processing_result & result, agent_resource_store & store,
        const agent_resource_read_authority & authority, const std::string & expected_mime,
        size_t max_bytes, agent_resource_processing_output & output, std::string & error) {
    output={};
    if(!result.success){error=result.failure_code+": "+result.safe_summary;return false;}
    if(result.outputs.size()==1 && result.resources.empty())output=result.outputs.front();
    else if(result.outputs.empty() && result.resources.size()==1){
        agent_resource_descriptor descriptor;
        if(!store.stat(result.resources.front().uri,authority,descriptor,error))return false;
        if(descriptor.size_bytes>max_bytes){error="processor resource exceeds the output limit";return false;}
        output.name=descriptor.name;output.mime_type=descriptor.mime_type;
        output.metadata=descriptor.metadata;output.lineage=descriptor.lineage;
        if(!store.read_bytes(descriptor.uri,authority,max_bytes+1,output.bytes,error))return false;
    }else{error="processor must return exactly one output or persisted resource";return false;}
    if(output.bytes.size()>max_bytes || common_normalize_resource_media_type(output.mime_type)!=common_normalize_resource_media_type(expected_mime)){
        error="processor returned an unexpected media type or oversized output";return false;
    }
    error.clear();return true;
}
