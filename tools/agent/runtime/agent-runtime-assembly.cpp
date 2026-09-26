#include "agent-runtime-assembly.h"

#include "../cli/agent-cli-runtime.h"
#include "../tooling/agent-tool-runtime-adapter.h"
#include "../adaptation/agent-learning-lifecycle-store.h"

#include <algorithm>

common_agent_inference_options make_agent_inference_options(common_agent_inference_options config) {
    return config;
}

common_agent_runtime_config make_agent_runtime_config(common_agent_runtime_build_config build_config) {
    common_agent_runtime_config config;
    config.generation_config = std::move(build_config.generation_config);
    config.context_budgets = build_config.context_budgets;
    config.generation_config.context_budgets = config.context_budgets;
    config.max_continuations = build_config.max_continuations;
    config.context_token_estimator = std::move(build_config.context_token_estimator);
    config.enable_memory_learning = build_config.enable_memory_learning;
    config.memory_learning_config = std::move(build_config.memory_learning_config);
    config.embed_memory = std::move(build_config.embed_memory);
    config.enable_adaptation_capture = build_config.enable_adaptation_capture;
    config.adaptation_config = std::move(build_config.adaptation_config);
    config.adaptation_transaction_backend = std::move(build_config.adaptation_transaction_backend);
    config.adaptation_transaction_path = std::move(build_config.adaptation_transaction_path);
    config.enable_flydelta_candidate_lifecycle = build_config.enable_flydelta_candidate_lifecycle;
    config.flydelta_lifecycle_backend = std::move(build_config.flydelta_lifecycle_backend);
    config.flydelta_lifecycle_path = std::move(build_config.flydelta_lifecycle_path);
    config.enable_flydelta_capture_candidates = build_config.enable_flydelta_capture_candidates;
    config.flydelta_model_profile_fingerprint = std::move(build_config.flydelta_model_profile_fingerprint);
    config.flydelta_capture_layout_revision = std::move(build_config.flydelta_capture_layout_revision);
    config.flydelta_max_capture_candidates = build_config.flydelta_max_capture_candidates;
    config.flydelta_capture_job_enqueue = std::move(build_config.flydelta_capture_job_enqueue);
    config.procedure_teaching_request_provider = std::move(build_config.procedure_teaching_request_provider);
    config.user_correction_teaching_request_provider = std::move(build_config.user_correction_teaching_request_provider);
    config.user_taught_concept_relation_provider = std::move(build_config.user_taught_concept_relation_provider);
    config.semantic_concept_hypothesis_provider = std::move(build_config.semantic_concept_hypothesis_provider);
    config.semantic_concept_hypothesis_batch_provider = std::move(build_config.semantic_concept_hypothesis_batch_provider);
    config.semantic_concept_grounding_provider = std::move(build_config.semantic_concept_grounding_provider);
    config.concept_candidate_index = std::move(build_config.concept_candidate_index);
    config.flydelta_teaching_material_observer = std::move(build_config.flydelta_teaching_material_observer);
    config.flydelta_teaching_material_runtime = std::move(build_config.flydelta_teaching_material_runtime);
    return config;
}

bool parse_agent_inference_backend(const std::string & value, agent_inference_backend & backend) {
    if (value == "cli") {
        backend = agent_inference_backend::cli;
        return true;
    }
    if (value == "server-context") {
        backend = agent_inference_backend::server_context;
        return true;
    }
    return false;
}

common_agent_runtime_assembly make_agent_runtime_assembly(
    common_memory_store & memory_store,
    common_plan_store & plan_store,
    common_agent_inference & inference,
    const common_agent_runtime_config & runtime_config,
    const std::vector<common_chat_tool> & tools,
    agent_tool_view * tool_view) {
    common_agent_runtime_assembly assembly;
    assembly.planner = make_llama_cli_planner(inference, runtime_config.generation_config, tools);
    assembly.executor = make_llama_cli_action_executor(inference, runtime_config.generation_config);
    assembly.reflector = make_llama_cli_reflection_engine(
        inference, runtime_config.generation_config, tools);

    if (runtime_config.enable_memory_learning) {
        assembly.candidate_extractor = make_llama_cli_memory_candidate_extractor(inference, runtime_config.generation_config);
        assembly.memory_learner = std::make_unique<common_memory_post_turn_learner>(
            memory_store,
            *assembly.candidate_extractor,
            runtime_config.embed_memory,
            runtime_config.memory_learning_config);
    }

    if (tool_view != nullptr) {
        assembly.tool_runtime = make_provider_agent_tool_runtime(*tool_view);
    }

    if (runtime_config.enable_adaptation_capture) {
        std::string store_error;
        assembly.adaptation_store = make_agent_learning_transaction_store(
            runtime_config.adaptation_transaction_backend,
            runtime_config.adaptation_transaction_path,
            store_error);
        if (!assembly.adaptation_store) assembly.adaptation_error = std::move(store_error);
        if (assembly.adaptation_store) {
            auto adaptation_config = runtime_config.adaptation_config;
            const bool has_teaching_relation_path =
                static_cast<bool>(runtime_config.procedure_teaching_request_provider) ||
                static_cast<bool>(runtime_config.user_correction_teaching_request_provider) ||
                static_cast<bool>(runtime_config.user_taught_concept_relation_provider) ||
                static_cast<bool>(runtime_config.semantic_concept_hypothesis_provider) ||
                static_cast<bool>(runtime_config.semantic_concept_hypothesis_batch_provider) ||
                static_cast<bool>(runtime_config.semantic_concept_grounding_provider) ||
                static_cast<bool>(runtime_config.flydelta_teaching_material_observer);
            const bool has_semantic_path =
                (static_cast<bool>(runtime_config.semantic_concept_hypothesis_provider) ||
                 static_cast<bool>(runtime_config.semantic_concept_hypothesis_batch_provider)) &&
                static_cast<bool>(runtime_config.semantic_concept_grounding_provider);
            if (!runtime_config.concept_candidate_index && has_semantic_path) {
                const bool need_lifecycle = runtime_config.enable_flydelta_candidate_lifecycle;
                if (need_lifecycle && !assembly.flydelta_lifecycle_store) {
                    std::string lifecycle_error;
                    assembly.flydelta_lifecycle_store = make_agent_learning_lifecycle_store(
                        runtime_config.flydelta_lifecycle_backend,
                        runtime_config.flydelta_lifecycle_path,
                        lifecycle_error);
                    if (!assembly.flydelta_lifecycle_store) assembly.adaptation_error = std::move(lifecycle_error);
                }
                assembly.concept_candidate_index = std::make_unique<common_agent_concept_candidate_index>(
                    assembly.flydelta_lifecycle_store.get());
                std::string candidate_error;
                if (!assembly.concept_candidate_index->load(candidate_error) && assembly.adaptation_error.empty()) {
                    assembly.adaptation_error = std::move(candidate_error);
                }
            }
            auto concept_candidate_index = runtime_config.concept_candidate_index;
            if (!concept_candidate_index) {
                concept_candidate_index = std::shared_ptr<common_agent_concept_candidate_index>(
                    assembly.concept_candidate_index.get(), [](common_agent_concept_candidate_index *) {});
            }
            if (runtime_config.enable_flydelta_capture_candidates || has_teaching_relation_path) {
                if (runtime_config.enable_flydelta_capture_candidates) {
                assembly.flydelta_capture_collector = std::make_unique<common_flydelta_capture_candidate_collector>(
                    runtime_config.flydelta_model_profile_fingerprint,
                    runtime_config.flydelta_capture_layout_revision,
                    runtime_config.flydelta_max_capture_candidates);
                if (runtime_config.enable_flydelta_candidate_lifecycle && !assembly.flydelta_lifecycle_store) {
                    std::string lifecycle_error;
                    assembly.flydelta_lifecycle_store = make_agent_learning_lifecycle_store(
                        runtime_config.flydelta_lifecycle_backend,
                        runtime_config.flydelta_lifecycle_path,
                        lifecycle_error);
                    if (!assembly.flydelta_lifecycle_store) {
                        assembly.adaptation_error = std::move(lifecycle_error);
                    }
                }
                if (assembly.flydelta_lifecycle_store || runtime_config.flydelta_capture_job_enqueue) {
                    assembly.flydelta_runtime_candidate_observer =
                        std::make_unique<common_flydelta_runtime_candidate_observer>(
                            *assembly.flydelta_capture_collector,
                            assembly.flydelta_lifecycle_store.get(),
                            runtime_config.flydelta_capture_job_enqueue);
                    adaptation_config.source_observer =
                        assembly.flydelta_runtime_candidate_observer->source_observer();
                } else {
                    adaptation_config.source_observer = assembly.flydelta_capture_collector->source_observer();
                }
                }
                auto teaching_material_observer = runtime_config.flydelta_teaching_material_observer;
                if (!teaching_material_observer && runtime_config.flydelta_teaching_material_runtime) {
                    const auto material_runtime = runtime_config.flydelta_teaching_material_runtime;
                    teaching_material_observer = [material_runtime](
                            const common_flydelta_teaching_relation & relation,
                            std::string & error) {
                        return material_runtime->observe_relation(relation, error);
                    };
                }
                if (runtime_config.procedure_teaching_request_provider) {
                    const auto provider = runtime_config.procedure_teaching_request_provider;
                    auto configured_relation_observer = adaptation_config.host_relation_observer;
                    auto * runtime_candidate_observer = assembly.flydelta_runtime_candidate_observer.get();
                    auto * capture_collector = assembly.flydelta_capture_collector.get();
                    adaptation_config.host_relation_observer =
                        [provider, configured_relation_observer, teaching_material_observer,
                            runtime_candidate_observer, capture_collector](
                                const common_agent_request & request,
                                const common_plan_state & plan,
                                const common_agent_result & result,
                                const common_learning_transaction & transaction,
                                std::string & error) {
                            if (configured_relation_observer &&
                                    !configured_relation_observer(request, plan, result, transaction, error)) return false;
                            std::optional<common_agent_procedure_teaching_request> request_value;
                            if (!provider(request, plan, result, transaction, request_value, error)) return false;
                            if (!request_value) return true;
                            const auto built = common_agent_build_procedure_teaching_relation(*request_value);
                            if (!built.relation) return true;
                            if (teaching_material_observer &&
                                    !teaching_material_observer(*built.relation, error)) return false;

                            common_adaptation_evidence_relation relation;
                            if (!common_agent_teaching_relation_to_evidence_relation(
                                    *built.relation, transaction, relation, error)) return false;

                            common_adaptation_evidence evidence;
                            if (!common_adaptation_evidence_from_turn(
                                    request, plan, result, relation, evidence, error)) return false;
                            if (runtime_candidate_observer) {
                                return runtime_candidate_observer->observe_verified_relation(
                                    relation, evidence, transaction, error);
                            }
                            if (capture_collector) {
                                return capture_collector->observe_verified_relation(
                                    relation, evidence, transaction, error);
                            }
                            return true;
                        };
                }
                if (runtime_config.user_correction_teaching_request_provider) {
                    const auto provider = runtime_config.user_correction_teaching_request_provider;
                    auto configured_relation_observer = adaptation_config.host_relation_observer;
                    auto * runtime_candidate_observer = assembly.flydelta_runtime_candidate_observer.get();
                    auto * capture_collector = assembly.flydelta_capture_collector.get();
                    adaptation_config.host_relation_observer =
                        [provider, configured_relation_observer, teaching_material_observer,
                            runtime_candidate_observer, capture_collector](
                                const common_agent_request & request,
                                const common_plan_state & plan,
                                const common_agent_result & result,
                                const common_learning_transaction & transaction,
                                std::string & error) {
                            if (configured_relation_observer &&
                                    !configured_relation_observer(request, plan, result, transaction, error)) return false;
                            std::optional<common_agent_user_correction_teaching_request> request_value;
                            if (!provider(request, plan, result, transaction, request_value, error)) return false;
                            if (!request_value) return true;
                            const auto built = common_agent_build_user_correction_teaching_relation(*request_value);
                            if (!built.relation) return true;
                            if (teaching_material_observer &&
                                    !teaching_material_observer(*built.relation, error)) return false;

                            common_adaptation_evidence_relation relation;
                            if (!common_agent_teaching_relation_to_evidence_relation(
                                    *built.relation, transaction, relation, error)) return false;
                            common_adaptation_evidence evidence;
                            if (!common_adaptation_evidence_from_turn(
                                    request, plan, result, relation, evidence, error)) return false;
                            if (runtime_candidate_observer) {
                                return runtime_candidate_observer->observe_verified_relation(
                                    relation, evidence, transaction, error);
                            }
                            if (capture_collector) {
                                return capture_collector->observe_verified_relation(
                                    relation, evidence, transaction, error);
                            }
                            return true;
                        };
                }
                if (runtime_config.user_taught_concept_relation_provider) {
                    const auto provider = runtime_config.user_taught_concept_relation_provider;
                    auto configured_relation_observer = adaptation_config.host_relation_observer;
                    auto * runtime_candidate_observer = assembly.flydelta_runtime_candidate_observer.get();
                    auto * capture_collector = assembly.flydelta_capture_collector.get();
                    adaptation_config.host_relation_observer =
                        [provider, configured_relation_observer, teaching_material_observer,
                            runtime_candidate_observer, capture_collector](
                                const common_agent_request & request,
                                const common_plan_state & plan,
                                const common_agent_result & result,
                                const common_learning_transaction & transaction,
                                std::string & error) {
                            if (configured_relation_observer &&
                                    !configured_relation_observer(request, plan, result, transaction, error)) return false;
                            std::vector<common_flydelta_teaching_relation> relations;
                            if (!provider(request, plan, result, transaction, relations, error)) return false;
                            for (const auto & teaching_relation : relations) {
                                if (teaching_relation.source != common_adaptation_evidence_source::user_taught_concept ||
                                        teaching_relation.status != common_flydelta_teaching_relation_status::resolved ||
                                        !teaching_relation.host_approved || teaching_relation.contrast_ref.empty()) {
                                    error = "user-taught concept provider returned an unresolved or ungrounded relation";
                                    return false;
                                }
                                if (teaching_material_observer &&
                                        !teaching_material_observer(teaching_relation, error)) return false;
                                common_adaptation_evidence_relation relation;
                                if (!common_agent_teaching_relation_to_evidence_relation(
                                        teaching_relation, transaction, relation, error)) return false;
                                common_adaptation_evidence evidence;
                                if (!common_adaptation_evidence_from_turn(
                                        request, plan, result, relation, evidence, error)) return false;
                                if (runtime_candidate_observer) {
                                    if (!runtime_candidate_observer->observe_verified_relation(
                                            relation, evidence, transaction, error)) return false;
                                } else if (capture_collector) {
                                    if (!capture_collector->observe_verified_relation(
                                            relation, evidence, transaction, error)) return false;
                                }
                            }
                            return true;
                        };
                }
                if ((runtime_config.semantic_concept_hypothesis_provider ||
                        runtime_config.semantic_concept_hypothesis_batch_provider) &&
                        runtime_config.semantic_concept_grounding_provider) {
                    const auto hypothesis_provider = runtime_config.semantic_concept_hypothesis_provider;
                    const auto hypothesis_batch_provider = runtime_config.semantic_concept_hypothesis_batch_provider;
                    const auto grounding_provider = runtime_config.semantic_concept_grounding_provider;
                    auto configured_relation_observer = adaptation_config.host_relation_observer;
                    auto * runtime_candidate_observer = assembly.flydelta_runtime_candidate_observer.get();
                    auto * capture_collector = assembly.flydelta_capture_collector.get();
                    adaptation_config.host_relation_observer =
                        [hypothesis_provider, grounding_provider, configured_relation_observer,
                            hypothesis_batch_provider, teaching_material_observer, runtime_candidate_observer,
                            capture_collector, concept_candidate_index](
                            const common_agent_request & request,
                            const common_plan_state & plan,
                            const common_agent_result & result,
                            const common_learning_transaction & transaction,
                            std::string & error) {
                            if (configured_relation_observer &&
                                    !configured_relation_observer(request, plan, result, transaction, error)) return false;
                            std::vector<common_agent_concept_hypothesis> hypotheses;
                            if (hypothesis_batch_provider) {
                                if (!hypothesis_batch_provider(request, plan, result, transaction, hypotheses, error)) return false;
                            } else {
                                std::optional<common_agent_concept_hypothesis> hypothesis;
                                if (!hypothesis_provider(request, plan, result, transaction, hypothesis, error)) return false;
                                if (hypothesis) hypotheses.push_back(std::move(*hypothesis));
                            }
                            for (const auto & hypothesis : hypotheses) {
                                if (!common_agent_concept_hypothesis_validate(hypothesis, error)) return false;
                                if (concept_candidate_index &&
                                        !concept_candidate_index->observe_hypothesis(hypothesis, transaction, error)) return false;
                                if (hypothesis.status != common_agent_concept_hypothesis_status::grounded ||
                                        !hypothesis.host_grounded || !hypothesis.reusable) continue;
                                std::vector<common_flydelta_teaching_relation> relations;
                                common_agent_concept_grounding grounding;
                                if (!grounding_provider(request, plan, result, transaction,
                                        hypothesis, grounding, relations, error)) return false;
                                for (const auto & relation : relations) {
                                    if (!common_agent_validate_concept_teaching_relation(
                                            hypothesis, grounding, relation, error)) return false;
                                    if (teaching_material_observer &&
                                            !teaching_material_observer(relation, error)) return false;
                                    if (concept_candidate_index &&
                                            !concept_candidate_index->observe_relation(
                                                hypothesis, relation, transaction, error)) return false;
                                    common_adaptation_evidence_relation evidence_relation;
                                    if (!common_agent_teaching_relation_to_evidence_relation(
                                            relation, transaction, evidence_relation, error)) return false;
                                    common_adaptation_evidence evidence;
                                    if (!common_adaptation_evidence_from_turn(
                                            request, plan, result, evidence_relation, evidence, error)) return false;
                                    if (runtime_candidate_observer) {
                                        if (!runtime_candidate_observer->observe_verified_relation(
                                                evidence_relation, evidence, transaction, error)) return false;
                                    } else if (capture_collector) {
                                        if (!capture_collector->observe_verified_relation(
                                                evidence_relation, evidence, transaction, error)) return false;
                                    }
                                }
                            }
                            return true;
                        };
                }
            }
            assembly.adaptation_observer = std::make_unique<common_learning_transaction_observer>(
                *assembly.adaptation_store, std::move(adaptation_config));
        }
    }

    assembly.runtime = std::make_unique<common_agent_runtime>(
        plan_store,
        *assembly.planner,
        *assembly.executor,
        *assembly.reflector,
        assembly.tool_runtime.get(),
        assembly.memory_learner.get(),
        nullptr,
        runtime_config.context_budgets,
        runtime_config.generation_config.context_size_tokens,
        static_cast<size_t>(std::max(0, runtime_config.generation_config.n_predict)),
        runtime_config.context_token_estimator);
    if (assembly.adaptation_observer) {
        assembly.runtime->set_adaptation_observer(assembly.adaptation_observer.get());
    }
    return assembly;
}
