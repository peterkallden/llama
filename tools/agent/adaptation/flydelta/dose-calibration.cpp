#include "../../daemon/agent-daemon-flydelta-internal.h"

namespace agent_daemon_flydelta_internal {
namespace {
constexpr const char * intervention_site = "layer-input:additive-cvec:v1";
constexpr const char * runtime_layout = "resident-cvec-runtime:v1";

std::string runtime_fingerprint(const daemon_flydelta_resource_provider & provider) {
    if (!provider.host) return {};
    const auto & params = provider.host->params();
    const auto & host = provider.host->config();
    const auto value = json{{"schema_version", 1}, {"model", provider.model_profile_fingerprint},
        {"n_ctx", params.n_ctx}, {"n_batch", params.n_batch}, {"n_ubatch", params.n_ubatch},
        {"n_parallel", params.n_parallel}, {"n_sequences", params.n_sequences},
        {"n_gpu_layers", params.n_gpu_layers}, {"fit_params", params.fit_params},
        {"cache_type_k", static_cast<int>(params.cache_type_k)}, {"cache_type_v", static_cast<int>(params.cache_type_v)},
        {"flash_attn", static_cast<int>(params.flash_attn_type)},
        {"per_sequence_cvec", host.per_sequence_cvec_batch},
        {"per_sequence_factorized_cvec", host.per_sequence_factorized_cvec_batch},
        {"layout", runtime_layout}}.dump();
    return "sha256:" + hash_sha256_hex(value.data(), value.size());
}

json calibration_json(const common_flydelta_residual_calibration & c, const std::string & input_ref) {
    return {{"kind", "flydelta_residual_calibration"}, {"schema_version", c.schema_version},
        {"input_ref", input_ref}, {"model_profile_fingerprint", c.model_profile_fingerprint},
        {"tokenizer_fingerprint", c.tokenizer_fingerprint}, {"template_fingerprint", c.template_fingerprint},
        {"capture_layout_revision", c.capture_layout_revision}, {"token_selection", c.token_selection},
        {"position", common_flydelta_capture_position_name(c.position)}, {"token_index", c.token_index},
        {"layer_index", c.layer_index}, {"dimension", c.dimension}, {"sample_norms", c.sample_norms},
        {"sample_count", c.sample_count}, {"median_norm", c.median_norm},
        {"intervention_site", c.intervention_site}, {"measurement_stage", c.measurement_stage},
        {"runtime_configuration_fingerprint", c.runtime_configuration_fingerprint},
        {"runtime_layout_revision", c.runtime_layout_revision}, {"dataset_revision", c.dataset_revision},
        {"aggregation", c.aggregation}, {"sequence_ids", c.sequence_ids}};
}
}

bool daemon_flydelta_load_dose_calibration(
        const std::shared_ptr<daemon_flydelta_resource_provider> & provider,
        const common_flydelta_experiment_fixture & fixture, int32_t layer_index,
        std::string & calibration_ref, common_flydelta_coefficient_search_config & config,
        std::string & content_fingerprint, std::string & error) {
    error.clear();
    content_fingerprint.clear();
    if (!provider || calibration_ref.empty() || !provider->host) {
        error = "FlyDelta direction_dose requires an authorized host calibration reference and resident host";
        return false;
    }
    config.injection_capture_layout_revision = provider->capture_layout_revision;
    config.injection_token_selection = "generation_boundary";
    config.injection_capture_position = common_flydelta_capture_position::generation_boundary;
    config.injection_intervention_site = intervention_site;
    config.injection_runtime_layout_revision = runtime_layout;
    config.injection_runtime_configuration_fingerprint = runtime_fingerprint(*provider);
    json value;
    if (!daemon_flydelta_read_json(*provider, calibration_ref, value, error)) return false;
    try {
        // An explicitly configured input manifest is a host-approved selection
        // of independent, already captured, unsteered calibration sequences.
        // It never launches inference or borrows the evaluation fixture.
        if (value.at("kind") == "flydelta_residual_calibration_inputs") {
            if (value.at("schema_version") != 1 || value.at("layer_index") != layer_index ||
                    value.at("model_profile_fingerprint") != fixture.model_profile_fingerprint ||
                    value.at("tokenizer_fingerprint") != fixture.tokenizer_fingerprint ||
                    value.at("template_fingerprint") != fixture.template_fingerprint ||
                    value.at("capture_layout_revision") != provider->capture_layout_revision ||
                    value.at("runtime_configuration_fingerprint") != config.injection_runtime_configuration_fingerprint ||
                    value.at("runtime_layout_revision") != runtime_layout ||
                    value.at("intervention_site") != intervention_site ||
                    value.at("measurement_stage") != "final_prompt_row" ||
                    !value.at("unsteered").get<bool>()) {
                error = "FlyDelta calibration input identity is incompatible"; return false;
            }
            const auto & entries = value.at("sequences");
            if (!entries.is_array() || entries.empty() || entries.size() > 4096) {
                error = "FlyDelta calibration sequences exceed their bound"; return false;
            }
            std::set<std::string> capture_refs;
            std::vector<std::string> ids;
            std::vector<std::vector<common_flydelta_hidden_state_capture>> sequences;
            for (const auto & entry : entries) {
                const auto reference = entry.at("capture_ref").get<std::string>();
                if (reference == fixture.id || !capture_refs.insert(reference).second) {
                    error = "FlyDelta calibration contains repeated or evaluation capture references"; return false;
                }
                json capture_json;
                if (!daemon_flydelta_read_json_bounded(*provider, reference, 16U * 1024U * 1024U, capture_json, error)) return false;
                if (capture_json.at("kind") != "flydelta_hidden_state_capture") {
                    error = "FlyDelta calibration input is not a hidden-state capture"; return false;
                }
                common_flydelta_hidden_state_capture capture;
                if (!daemon_flydelta_capture_from_json(capture_json, capture, error)) return false;
                if (capture.position != common_flydelta_capture_position::generation_boundary) {
                    error = "FlyDelta sequence calibration requires a final prompt-row capture"; return false;
                }
                ids.push_back(entry.at("sequence_id").get<std::string>());
                sequences.push_back({std::move(capture)});
            }
            common_flydelta_residual_calibration c;
            if (!common_flydelta_calibrate_residual_sequences(sequences, ids, layer_index, calibration_ref,
                    "generation_boundary", c, error)) return false;
            c.tokenizer_fingerprint = fixture.tokenizer_fingerprint;
            c.template_fingerprint = fixture.template_fingerprint;
            c.dataset_revision = value.at("dataset_revision").get<std::string>();
            c.intervention_site = intervention_site;
            c.runtime_layout_revision = runtime_layout;
            c.runtime_configuration_fingerprint = config.injection_runtime_configuration_fingerprint;
            value = calibration_json(c, calibration_ref);
            const auto text = value.dump();
            const auto digest = hash_sha256_hex(text.data(), text.size());
            if (!daemon_flydelta_put_json_resource(provider, "flydelta-calibration-" + digest.substr(0, 24) + ".json",
                    "flydelta_residual_calibration", calibration_ref, text, calibration_ref, error)) return false;
            json persisted;
            if (!daemon_flydelta_read_json(*provider, calibration_ref, persisted, error) || persisted != value) {
                if (error.empty()) error = "FlyDelta calibration failed persistence verification";
                return false;
            }
        }
        if (value.at("kind") != "flydelta_residual_calibration" || value.at("schema_version") != 2) {
            error = "FlyDelta direction_dose requires calibration schema version 2"; return false;
        }
        auto & c = config.residual_calibration;
        c.schema_version = 2;
        c.reference = calibration_ref;
        c.model_profile_fingerprint = value.at("model_profile_fingerprint").get<std::string>();
        c.tokenizer_fingerprint = value.at("tokenizer_fingerprint").get<std::string>();
        c.template_fingerprint = value.at("template_fingerprint").get<std::string>();
        c.capture_layout_revision = value.at("capture_layout_revision").get<std::string>();
        c.token_selection = value.at("token_selection").get<std::string>();
        if (value.at("position") != "generation_boundary") { error = "FlyDelta calibration capture position differs"; return false; }
        c.position = common_flydelta_capture_position::generation_boundary;
        c.token_index = value.at("token_index").get<int32_t>();
        c.layer_index = value.at("layer_index").get<int32_t>();
        c.dimension = value.at("dimension").get<size_t>();
        c.sample_norms = value.at("sample_norms").get<std::vector<float>>();
        c.sample_count = value.at("sample_count").get<size_t>();
        c.median_norm = value.at("median_norm").get<float>();
        c.intervention_site = value.at("intervention_site").get<std::string>();
        c.measurement_stage = value.at("measurement_stage").get<std::string>();
        c.runtime_configuration_fingerprint = value.at("runtime_configuration_fingerprint").get<std::string>();
        c.runtime_layout_revision = value.at("runtime_layout_revision").get<std::string>();
        c.dataset_revision = value.at("dataset_revision").get<std::string>();
        c.aggregation = value.at("aggregation").get<std::string>();
        c.sequence_ids = value.at("sequence_ids").get<std::vector<std::string>>();
        config.injection_token_index = c.token_index;
        // Run the same identity/numerical validation used by actual proposal
        // generation before any model execution. This one-vector probe is
        // host memory only; it never evaluates a model or publishes an arm.
        common_flydelta_low_rank_basis probe;
        probe.layer_index = layer_index;
        probe.dimension = provider->model_n_embd;
        if (probe.dimension == 0 || probe.dimension > 1048576) { error = "FlyDelta model dimension is invalid"; return false; }
        probe.vectors = {std::vector<float>(probe.dimension, 0.0f)};
        probe.vectors.front().front() = 1.0f;
        std::vector<std::vector<float>> ignored;
        if (!common_flydelta_propose_calibrated_coefficients(fixture, probe, config, ignored, error)) return false;
        const auto text = value.dump();
        content_fingerprint = hash_sha256_hex(text.data(), text.size());
        return true;
    } catch (const std::exception & exception) {
        error = std::string("FlyDelta dose calibration is malformed: ") + exception.what(); return false;
    }
}
} // namespace agent_daemon_flydelta_internal
