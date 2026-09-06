#include "astc-vulkan-model-cache.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

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

    // Usage/benefit planning is a separate, non-serialized layer. It can
    // reorder validated artifacts and optionally demote zero-benefit entries
    // to native fallback without changing their manifest records.
    astc_vulkan_model_cache_usage_options usage_options;
    std::vector<astc_vulkan_tensor_usage_metrics> usage = {
        {"blk.0.ffn_down.weight", 10, 10, 1000, 400, 100.0, 60.0, 1.0, 2},
        {"blk.2.ffn_down.weight", 2, 2, 100, 100, 20.0, 20.0, 1.0, 1},
    };
    astc_vulkan_model_cache_plan usage_plan;
    assert(astc_vulkan_model_cache_plan_usage(
        experimental_plan, usage, usage_options, budget, usage_plan, error));
    assert(usage_plan.entries.size() == experimental_plan.entries.size());
    assert(usage_plan.entries.front().tensor_name == "blk.0.ffn_down.weight");
    assert(usage_plan.entries.front().usage_available);
    assert(usage_plan.entries.front().benefit_score > 0.0);

    usage_options.require_positive_benefit = true;
    astc_vulkan_model_cache_plan gated_plan;
    assert(astc_vulkan_model_cache_plan_usage(
        experimental_plan, usage, usage_options, budget, gated_plan, error));
    bool found_zero_benefit_fallback = false;
    for (const auto & entry : gated_plan.entries) {
        if (entry.tensor_name == "blk.2.ffn_down.weight") {
            found_zero_benefit_fallback = entry.use_native_fallback;
        }
    }
    assert(found_zero_benefit_fallback);

    std::vector<astc_vulkan_model_cache_storage_page> pages;
    assert(astc_vulkan_model_cache_make_storage_pages(
        experimental_plan, 0, pages, error));
    assert(pages.size() == 2);
    assert(pages[0].payload_bytes != 0 && pages[1].payload_bytes != 0);

    const auto usage_path = std::filesystem::temp_directory_path() /
        "astc-vulkan-usage-metrics-contract.txt";
    std::filesystem::remove(usage_path);
    assert(astc_vulkan_write_tensor_usage_metrics(usage_path.string(), usage, error));
    std::vector<astc_vulkan_tensor_usage_metrics> loaded_usage;
    assert(astc_vulkan_read_tensor_usage_metrics(usage_path.string(), loaded_usage, error));
    assert(loaded_usage.size() == usage.size());
    assert(loaded_usage[0].tensor_name == usage[0].tensor_name);
    assert(loaded_usage[0].native_gpu_time_ns == usage[0].native_gpu_time_ns);
    std::filesystem::remove(usage_path);

    const auto malformed_usage_path = std::filesystem::temp_directory_path() /
        "astc-vulkan-usage-metrics-malformed.txt";
    {
        std::ofstream malformed(malformed_usage_path);
        malformed << "# astc-usage-v1\n"
                  << "blk.0.ffn_down.weight 1 1 1 1 1 1 1.0 0\n"
                  << "blk.0.ffn_down.weight 2 2 2 2 2 2 1.0 1\n";
    }
    loaded_usage.clear();
    assert(!astc_vulkan_read_tensor_usage_metrics(
        malformed_usage_path.string(), loaded_usage, error));
    std::filesystem::remove(malformed_usage_path);

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

    const auto merge_root = std::filesystem::temp_directory_path() /
        "astc-vulkan-model-cache-merge-contract";
    std::filesystem::remove_all(merge_root);
    std::filesystem::create_directories(merge_root);
    std::vector<astc_vulkan_model_cache_fragment> fragments;
    for (int index = 0; index < 2; ++index) {
        const auto fragment_root = merge_root / ("fragment-" + std::to_string(index));
        std::filesystem::create_directories(fragment_root);
        const auto payload_path = fragment_root / "payload.astcpack";
        const auto manifest_path = fragment_root / "manifest.astcv";
        std::vector<uint8_t> bytes(16, static_cast<uint8_t>(index + 1));
        {
            std::ofstream payload(payload_path, std::ios::binary);
            payload.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
        }
        astc_vulkan_artifact_record artifact = make_artifact(
            index == 0 ? "fragment-a" : "fragment-b",
            index == 0 ? "blk.0.ffn_down.weight" : "blk.1.ffn_down.weight",
            0.1f, true, true);
        artifact.storage.payload_hash64 = astc_vulkan_payload_hash64(bytes.data(), bytes.size());
        astc_vulkan_manifest fragment_manifest;
        fragment_manifest.version = 4;
        fragment_manifest.model_fingerprint = "model-test";
        fragment_manifest.artifacts.push_back(artifact);
        assert(astc_vulkan_write_manifest(manifest_path.string(), fragment_manifest, error));
        fragments.push_back({manifest_path.string(), payload_path.string(), {}, {}});
    }
    const auto merged_manifest_path = merge_root / "merged.manifest.astcv";
    const auto merged_payload_path = merge_root / "merged.payload.astcpack";
    astc_vulkan_manifest merged;
    assert(astc_vulkan_model_cache_merge_fragments(
        fragments, merged_manifest_path.string(), merged_payload_path.string(), {}, {}, merged, error));
    assert(merged.artifacts.size() == 2);
    assert(merged.artifacts[0].storage.byte_offset == 0);
    assert(merged.artifacts[1].storage.byte_offset == 16);
    assert(std::filesystem::file_size(merged_payload_path) == 32);
    astc_vulkan_manifest reloaded;
    assert(astc_vulkan_read_manifest(merged_manifest_path.string(), reloaded, error));
    assert(reloaded.artifacts.size() == merged.artifacts.size());
    std::filesystem::remove_all(merge_root);

    std::cout << "ASTC Vulkan model cache plan contract passed\n";
    return 0;
}
