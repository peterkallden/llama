#include "astc-gpu-encoder-finisher.h"

#include <astcenc.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

static bool astc_gpu_encoder_finish_parallel(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const astc_gpu_encoder_finish_options & options,
    const astc_vulkan_format_info & format,
    const astcenc_config & config,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error);

bool astc_gpu_encoder_finish_with_options(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const astc_gpu_encoder_finish_options & options,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    std::vector<astc_gpu_encoder_batch> batches;
    if (!astc_gpu_encoder_plan_batches(request, batches)) {
        error = "CPU finisher requires valid physical proposal input";
        return false;
    }
    const auto format = astc_vulkan_format(request.footprint);
    if (format.block_width == 0 || format.block_height == 0) {
        error = "CPU finisher received unsupported ASTC footprint";
        return false;
    }
    std::unordered_map<uint32_t, const astc_gpu_encoder_source_block *> source;
    source.reserve(request.blocks.size());
    for (const auto & block : request.blocks) {
        if (!source.emplace(block.source_block_id, &block).second) {
            error = "duplicate GPU source block id";
            return false;
        }
    }
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height, 1,
                            static_cast<float>(options.quality), 0, &config) != ASTCENC_SUCCESS) {
        error = "astcenc configuration failed in CPU finisher";
        return false;
    }
    if (options.worker_count > 1 && retained_proposals.size() > 1) {
        return astc_gpu_encoder_finish_parallel(
            request, retained_proposals, options, format, config, finished, error);
    }
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) {
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
#endif
        error = "astcenc context allocation failed in CPU finisher";
        return false;
    }
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    finished.clear();
    finished.reserve(retained_proposals.size());
    for (const auto & proposal : retained_proposals) {
        const auto found = source.find(proposal.source_block_id);
        if (found == source.end()) {
            astcenc_context_free(context);
            error = "GPU proposal references missing source block";
            return false;
        }
        std::vector<float> input(size_t(format.block_width) * format.block_height * 4);
        for (uint32_t texel = 0; texel < format.block_width * format.block_height; ++texel) for (uint32_t channel = 0; channel < 4; ++channel) {
            input[texel * 4 + channel] = found->second->texels[texel].rgba[channel];
        }
        void * input_slice = input.data();
        astcenc_image input_image{format.block_width, format.block_height, 1, ASTCENC_TYPE_F32, &input_slice};
        astc_gpu_encoder_finished_block result;
        result.source_block_id = proposal.source_block_id;
        result.footprint = request.footprint;
        if (astcenc_compress_image(context, &input_image, &swizzle, result.payload.data(),
                                   result.payload.size(), 0) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            error = "astcenc CPU finisher compression failed";
            return false;
        }
        astcenc_block_info info{};
        if (astcenc_get_block_info(context, result.payload.data(), &info) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            error = "CPU finisher emitted illegal ASTC payload";
            return false;
        }
        result.decoded_rgba.resize(input.size());
        void * decoded_slice = result.decoded_rgba.data();
        astcenc_image decoded_image{format.block_width, format.block_height, 1, ASTCENC_TYPE_F32, &decoded_slice};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &decoded_image, &swizzle, 0) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            error = "CPU finisher exact decode failed";
            return false;
        }
        finished.push_back(result);
    }
    astcenc_context_free(context);
    error.clear();
    return true;
}

bool astc_gpu_encoder_finish_neural_hybrid(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const std::vector<uint32_t> & mandatory_reference_source_ids,
    const astc_gpu_encoder_neural_hybrid_finish_options & options,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    finished.clear();
    if (mandatory_reference_source_ids.empty()) {
        error = "neural hybrid requires a mandatory reference per logical block";
        return false;
    }

    std::unordered_set<uint32_t> reference_ids;
    reference_ids.reserve(mandatory_reference_source_ids.size());
    std::vector<astc_gpu_encoder_proposal> references;
    references.reserve(mandatory_reference_source_ids.size());
    for (uint32_t source_block_id : mandatory_reference_source_ids) {
        if (!reference_ids.emplace(source_block_id).second) {
            error = "neural hybrid received duplicate mandatory reference id";
            return false;
        }
        astc_gpu_encoder_proposal reference;
        reference.source_block_id = source_block_id;
        references.push_back(reference);
    }

    std::vector<astc_gpu_encoder_finished_block> reference_finished;
    if (!astc_gpu_encoder_finish_with_options(
            request, references,
            {options.reference_quality, astc_gpu_encoder_finish_mode::reference,
             options.worker_count},
            reference_finished, error)) return false;

    // A reference source can also have survived proposal retention. Do not
    // encode it twice: the reference payload is intentionally authoritative.
    std::unordered_set<uint32_t> seen = reference_ids;
    std::vector<astc_gpu_encoder_proposal> exploration;
    exploration.reserve(retained_proposals.size());
    for (const auto & proposal : retained_proposals) {
        if (seen.emplace(proposal.source_block_id).second) exploration.push_back(proposal);
    }

    std::vector<astc_gpu_encoder_finished_block> exploration_finished;
    if (!exploration.empty() && !astc_gpu_encoder_finish_with_options(
            request, exploration,
            {options.exploration_quality, astc_gpu_encoder_finish_mode::guided,
             options.worker_count},
            exploration_finished, error)) return false;

    finished.reserve(reference_finished.size() + exploration_finished.size());
    finished.insert(finished.end(), reference_finished.begin(), reference_finished.end());
    finished.insert(finished.end(), exploration_finished.begin(), exploration_finished.end());
    error.clear();
    return true;
}

static bool astc_gpu_encoder_finish_parallel(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    const astc_gpu_encoder_finish_options & options,
    const astc_vulkan_format_info & format,
    const astcenc_config & config,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    std::unordered_map<uint32_t, const astc_gpu_encoder_source_block *> source;
    source.reserve(request.blocks.size());
    for (const auto & block : request.blocks) {
        if (!source.emplace(block.source_block_id, &block).second) {
            error = "duplicate GPU source block id";
            return false;
        }
    }
    const uint32_t worker_count = std::max(1u, std::min<uint32_t>(
        options.worker_count, static_cast<uint32_t>(retained_proposals.size())));
    std::vector<astcenc_context *> contexts(worker_count, nullptr);
    for (astcenc_context * & context : contexts) {
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
        if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) {
#else
        if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
#endif
            for (astcenc_context * allocated : contexts) if (allocated) astcenc_context_free(allocated);
            error = "astcenc context allocation failed in parallel CPU finisher";
            return false;
        }
    }

    finished.assign(retained_proposals.size(), {});
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    std::atomic<size_t> next_proposal{0};
    std::atomic<bool> failed{false};
    std::mutex error_mutex;
    std::string worker_error;
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (uint32_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&, worker]() {
            astcenc_context * context = contexts[worker];
            while (!failed.load(std::memory_order_relaxed)) {
                const size_t proposal_index = next_proposal.fetch_add(1, std::memory_order_relaxed);
                if (proposal_index >= retained_proposals.size()) break;
                const auto & proposal = retained_proposals[proposal_index];
                const auto found = source.find(proposal.source_block_id);
                if (found == source.end()) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (worker_error.empty()) worker_error = "GPU proposal references missing source block";
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                std::vector<float> input(size_t(format.block_width) * format.block_height * 4);
                for (uint32_t texel = 0; texel < format.block_width * format.block_height; ++texel) {
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        input[texel * 4 + channel] = found->second->texels[texel].rgba[channel];
                    }
                }
                void * input_slice = input.data();
                astcenc_image input_image{format.block_width, format.block_height, 1,
                                          ASTCENC_TYPE_F32, &input_slice};
                astc_gpu_encoder_finished_block result;
                result.source_block_id = proposal.source_block_id;
                result.footprint = request.footprint;
                if (astcenc_compress_image(context, &input_image, &swizzle, result.payload.data(),
                                           result.payload.size(), 0) != ASTCENC_SUCCESS) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (worker_error.empty()) worker_error = "astcenc CPU finisher compression failed";
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                astcenc_block_info info{};
                if (astcenc_get_block_info(context, result.payload.data(), &info) != ASTCENC_SUCCESS) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (worker_error.empty()) worker_error = "CPU finisher emitted illegal ASTC payload";
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                result.decoded_rgba.resize(input.size());
                void * decoded_slice = result.decoded_rgba.data();
                astcenc_image decoded_image{format.block_width, format.block_height, 1,
                                            ASTCENC_TYPE_F32, &decoded_slice};
                if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                             &decoded_image, &swizzle, 0) != ASTCENC_SUCCESS) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (worker_error.empty()) worker_error = "astcenc CPU finisher exact decode failed";
                    failed.store(true, std::memory_order_relaxed);
                    break;
                }
                finished[proposal_index] = std::move(result);
            }
        });
    }
    for (std::thread & worker : workers) worker.join();
    for (astcenc_context * context : contexts) if (context) astcenc_context_free(context);
    if (failed.load(std::memory_order_relaxed)) {
        error = worker_error.empty() ? "parallel CPU ASTC finisher failed" : worker_error;
        finished.clear();
        return false;
    }
    error.clear();
    return true;
}

bool astc_gpu_encoder_finish(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    return astc_gpu_encoder_finish_with_options(
        request, retained_proposals,
        astc_gpu_encoder_finish_options{quality, astc_gpu_encoder_finish_mode::reference},
        finished, error);
}

bool astc_gpu_encoder_finish_d1_scalar(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    return astc_gpu_encoder_finish(request, retained_proposals, quality, finished, error);
}

bool astc_gpu_encoder_finish_d1_scalar_4x4(
    const astc_gpu_encoder_request & request,
    const std::vector<astc_gpu_encoder_proposal> & retained_proposals,
    float quality,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    if (request.footprint != astc_vulkan_footprint::k4x4) {
        error = "v1 GPU proposer finisher requires D1 scalar 4x4";
        return false;
    }
    return astc_gpu_encoder_finish(request, retained_proposals, quality, finished, error);
}

bool astc_gpu_exact_subset_finish_payloads(
    astc_vulkan_footprint footprint,
    const std::vector<astc_gpu_exact_subset_block> & payloads,
    std::vector<astc_gpu_encoder_finished_block> & finished,
    std::string & error) {
    finished.clear();
    const auto format = astc_vulkan_format(footprint);
    if (payloads.empty() || format.block_width == 0 || format.block_height == 0) {
        error = "exact-subset decode requires non-empty supported payloads";
        return false;
    }
    astcenc_config config{};
    if (astcenc_config_init(ASTCENC_PRF_LDR, format.block_width, format.block_height,
                            1, ASTCENC_PRE_FAST, 0, &config) != ASTCENC_SUCCESS) {
        error = "exact-subset decode configuration failed";
        return false;
    }
    astcenc_context * context = nullptr;
#if defined(GGML_VK_ASTC_EXPERIMENTAL_NEURAL_RANK)
    if (astcenc_context_alloc(&config, 1, &context, nullptr) != ASTCENC_SUCCESS) {
#else
    if (astcenc_context_alloc(&config, 1, &context) != ASTCENC_SUCCESS) {
#endif
        error = "exact-subset decode context allocation failed";
        return false;
    }
    const astcenc_swizzle swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    finished.reserve(payloads.size());
    for (const auto & payload : payloads) {
        astcenc_block_info info{};
        if (astcenc_get_block_info(context, payload.payload.data(), &info) != ASTCENC_SUCCESS ||
            info.is_error_block) {
            astcenc_context_free(context);
            finished.clear();
            error = "exact-subset payload is not legal LDR ASTC";
            return false;
        }
        astc_gpu_encoder_finished_block result;
        result.source_block_id = payload.source_block_id;
        result.footprint = footprint;
        result.payload = payload.payload;
        result.decoded_rgba.resize(size_t(format.block_width) * format.block_height * 4);
        void * decoded = result.decoded_rgba.data();
        astcenc_image image{format.block_width, format.block_height, 1, ASTCENC_TYPE_F32, &decoded};
        if (astcenc_decompress_image(context, result.payload.data(), result.payload.size(),
                                     &image, &swizzle, 0) != ASTCENC_SUCCESS) {
            astcenc_context_free(context);
            finished.clear();
            error = "exact-subset payload CPU decode failed";
            return false;
        }
        finished.push_back(std::move(result));
    }
    astcenc_context_free(context);
    error.clear();
    return true;
}
