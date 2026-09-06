#include "astc-vulkan-model-cache.h"

#include <cassert>
#include <filesystem>
#include <iostream>

namespace {

astc_vulkan_artifact_record make_artifact(const char * id,
                                          const char * tensor,
                                          float loss_delta,
                                          bool model_gate,
                                          bool vulkan_gate) {
    astc_vulkan_artifact_record artifact;
    artifact.id = id;
    artifact.storage.name = tensor;
    artifact.storage.width = 6;
    artifact.storage.height = 6;
    artifact.storage.footprint = astc_vulkan_footprint::k6x6;
    artifact.storage.byte_size = astc_vulkan_image_bytes(
        artifact.storage.footprint, artifact.storage.width, artifact.storage.height);
    artifact.evidence.model_gate_passed = model_gate;
    artifact.evidence.vulkan_gate_passed = vulkan_gate;
    artifact.evidence.loss_delta = loss_delta;
    artifact.evidence.logits_relative_mse = loss_delta;
    artifact.evidence.top1_agreement = 1.0f;
    return artifact;
}

} // namespace

int main() {
    astc_vulkan_manifest manifest;
    manifest.version = 4;
    manifest.model_fingerprint = "model-test";
    manifest.artifacts.push_back(make_artifact(
        "layer0-neutral", "blk.0.ffn_down.weight", 0.20f, true, true));
    auto selected = make_artifact(
        "layer0-selected", "blk.0.ffn_down.weight", 0.10f, true, true);
    selected.variant = astc_vulkan_artifact_variant::validation_selected;
    manifest.artifacts.push_back(selected);
    manifest.artifacts.push_back(make_artifact(
        "layer1-unapproved", "blk.1.ffn_down.weight", 0.01f, false, true));
    auto experimental = make_artifact(
        "layer2-experimental", "blk.2.ffn_down.weight", 0.03f, true, true);
    experimental.storage.footprint = astc_vulkan_footprint::k8x5;
    manifest.artifacts.push_back(experimental);

    astc_vulkan_model_cache_plan_options options;
    options.policy = astc_vulkan_quality_policy::quality;
    astc_vulkan_model_cache_plan plan;
    std::string error;
    assert(astc_vulkan_model_cache_make_plan(manifest, options, plan, error));
    assert(plan.entries.size() == 3);
    assert(plan.entries[0].tensor_name == "blk.0.ffn_down.weight");
    assert(plan.entries[0].artifact_id == "layer0-selected");
    assert(!plan.entries[0].use_native_fallback);
    assert(plan.entries[1].tensor_name == "blk.1.ffn_down.weight");
    assert(plan.entries[1].use_native_fallback);
    assert(plan.entries[2].tensor_name == "blk.2.ffn_down.weight");
    assert(plan.entries[2].use_native_fallback);

    options.allow_experimental = true;
    astc_vulkan_model_cache_plan experimental_plan;
    assert(astc_vulkan_model_cache_make_plan(
        manifest, options, experimental_plan, error));
    assert(experimental_plan.entries[2].artifact_id == "layer2-experimental");

    astc_vulkan_memory_budget budget;
    budget.effective_device_limit_bytes = plan.entries[0].device_bytes;
    budget.host_limit_bytes = plan.entries[0].host_bytes;
    astc_vulkan_model_cache_plan resident_plan;
    assert(astc_vulkan_model_cache_plan_residency(
        plan, budget, resident_plan, error));
    assert(resident_plan.residency.resident_items.size() == 1);
    assert(resident_plan.residency.requires_streaming);

    const auto base = std::filesystem::temp_directory_path() /
        "astc-vulkan-model-cache-contract";
    std::filesystem::remove(base);
    std::filesystem::remove(base.string() + ".partial");
    astc_vulkan_model_cache_build_state state;
    state.source_model = "model-test.gguf";
    state.output_root = base.string();
    state.completed_tensors = {"blk.0.ffn_down.weight"};
    assert(astc_vulkan_model_cache_write_build_state(base.string(), state, error));
    astc_vulkan_model_cache_build_state loaded;
    assert(astc_vulkan_model_cache_read_build_state(base.string(), loaded, error));
    assert(loaded.source_model == state.source_model);
    assert(loaded.output_root == state.output_root);
    assert(loaded.completed_tensors == state.completed_tensors);
    std::filesystem::remove(base);
    std::filesystem::remove(base.string() + ".partial");

    std::cout << "ASTC Vulkan model cache plan contract passed\n";
    return 0;
}
