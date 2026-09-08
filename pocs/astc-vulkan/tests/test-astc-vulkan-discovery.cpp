#include "astc-vulkan-discovery.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    std::vector<astc_vulkan_discovery_tensor_input> tensors = {
        {"hot", 1024, 1024, 8u * 1024u * 1024u, true},
        {"cold", 1024, 1024, 8u * 1024u * 1024u, true},
        {"vector", 1024, 1, 4096, false},
    };
    std::vector<astc_vulkan_tensor_usage_metrics> usage = {
        {"hot", 100, 100, 8u * 1024u * 1024u, 0, 1000.0, 0.0, 1.0, 0},
        {"cold", 1, 1, 8u * 1024u * 1024u, 0, 1000.0, 0.0, 1.0, 1},
    };
    astc_vulkan_discovery_options options;
    options.footprint = astc_vulkan_footprint::k8x5;
    options.representation = astc_vulkan_representation::kPairedD2;
    options.max_tensors = 1;
    std::vector<astc_vulkan_discovery_entry> result;
    std::string error;
    assert(astc_vulkan_discover_cache_candidates(tensors, usage, options, result, error));
    assert(result.size() == 2);
    assert(result[0].tensor_name == "hot" && result[0].selected);
    assert(!result[1].selected);
    assert(result[0].estimated_layout_bytes != 0);
    const auto report = std::filesystem::temp_directory_path() / "astc-discovery-test.tsv";
    assert(astc_vulkan_write_discovery_report(report.string(), options, result, error));
    std::ifstream input(report);
    assert(input.good());
    std::filesystem::remove(report);

    // Discovery proposes a bounded representation-neutral bank. The high
    // range spread makes absmax conditional candidates worthwhile to test,
    // while D1 10x8 remains the iso-rate control rather than an afterthought.
    const std::vector<astc_vulkan_discovery_quality_probe> probes = {
        {"hot", true, "calibration-a", 3.0, .02, .01},
    };
    std::vector<astc_vulkan_discovery_candidate_plan_entry> plan;
    assert(astc_vulkan_make_discovery_candidate_plan(result, probes, options, {}, plan, error));
    assert(plan.size() == 5);
    assert(plan[0].kind == astc_vulkan_discovery_candidate_kind::d2_la_pairing_neutral);
    assert(plan[1].kind == astc_vulkan_discovery_candidate_kind::d2_la_pairing_selected);
    assert(plan[2].conditional && plan[3].conditional);
    assert(plan[4].kind == astc_vulkan_discovery_candidate_kind::d1_10x8);
    const auto candidate_plan = std::filesystem::temp_directory_path() / "astc-discovery-plan-test.tsv";
    assert(astc_vulkan_write_discovery_candidate_plan(candidate_plan.string(), plan, error));
    input.open(candidate_plan);
    assert(input.good());
    input.close();
    std::filesystem::remove(candidate_plan);
    return 0;
}
