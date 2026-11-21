#include "agent/visualization/visualization-contract.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace {
int failures = 0;
void check(bool value, const char * message) {
    if (!value) { std::cerr << message << '\n'; ++failures; }
}

}

int main() {
    std::string error;
    std::ifstream fixture(std::string(LLAMA_AGENT_SOURCE_DIR) + "/tests/fixtures/visualization-relationship-v1.json");
    std::stringstream buffer; buffer << fixture.rdbuf();
    const auto serialized = nlohmann::json::parse(buffer.str(), nullptr, false);
    common_agent_visualization_relationship decoded;
    check(common_agent_visualization_relationship_from_json(serialized, decoded, error) &&
        decoded.nodes.size() == 2 && decoded.edges[0].source == "svc-api" && decoded.data_revision == "fixture-r1",
        "shared relationship fixture must parse and preserve its graph revision");
    const auto roundtrip = common_agent_visualization_relationship_to_json(decoded);
    check(common_agent_visualization_relationship_from_json(roundtrip, decoded, error) &&
        decoded.nodes[1].label == "Storage service", "relationship serialization must roundtrip");

    auto invalid = decoded; invalid.edges[0].target = "missing";
    check(!common_agent_visualization_relationship_validate(invalid, error), "edge with missing endpoint must be rejected");
    invalid = decoded; invalid.edges[0].id = invalid.nodes[0].id;
    check(!common_agent_visualization_relationship_validate(invalid, error), "node and edge IDs share a unique namespace");
    invalid = decoded; invalid.view.selected_nodes = {"missing"};
    check(!common_agent_visualization_relationship_validate(invalid, error), "selection must refer to a node in the projection");
    invalid = decoded; invalid.complete = false;
    check(!common_agent_visualization_relationship_validate(invalid, error), "incomplete projection must explain a limitation");
    invalid = decoded; invalid.nodes.resize(COMMON_AGENT_VISUALIZATION_MAX_NODES + 1);
    check(!common_agent_visualization_relationship_validate(invalid, error), "node limit must be enforced");

    auto hostile = serialized;
    hostile.erase("view");
    check(common_agent_visualization_relationship_from_json(hostile, decoded, error),
        "omitted view must use the same defaults as the client contract");
    hostile = serialized; hostile["schema_version"] = 4294967297ULL;
    check(!common_agent_visualization_relationship_from_json(hostile, decoded, error),
        "large schema version must not wrap to version one");
    hostile = serialized; hostile["complete"] = true; hostile["truncated"] = true;
    hostile["limitations"] = {"bounded"};
    check(!common_agent_visualization_relationship_from_json(hostile, decoded, error),
        "truncated projection cannot claim completeness");
    hostile = serialized;
    hostile["nodes"][0]["html"] = "<script>run()</script>";
    check(!common_agent_visualization_relationship_from_json(hostile, decoded, error),
        "unknown node fields must be rejected");
    hostile = serialized; hostile["view"]["layout"] = "custom-code";
    check(!common_agent_visualization_relationship_from_json(hostile, decoded, error),
        "unknown renderer layouts must be rejected");
    return failures == 0 ? 0 : 1;
}
