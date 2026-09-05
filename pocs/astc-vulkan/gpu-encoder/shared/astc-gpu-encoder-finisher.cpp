#include "astc-gpu-encoder-finisher.h"

#include <astcenc.h>

#include <algorithm>
#include <unordered_map>

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
