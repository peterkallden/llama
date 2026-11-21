#pragma once

#include "agent/adaptation/semantic-teaching.h"

#include <functional>

// A host owns the semantic contrast.  This adapter only turns the existing
// host-verified contrast seam into the existing grounding provider seam; it
// does not create a second material/evidence store or infer correctness.
using common_agent_concept_contrast_provider = std::function<bool(
        const common_agent_request & request,
        const common_plan_state & plan,
        const common_agent_result & result,
        const common_learning_transaction & transaction,
        const common_agent_concept_hypothesis & hypothesis,
        std::vector<common_agent_concept_contrast> & contrasts,
        std::string & error)>;

// Research extraction is deliberately strict in V0.  A research evidence
// statement is admitted only when it is an explicit structured JSON concept
// envelope.  Free-form research prose remains research output, not learning
// input.
common_agent_concept_hypothesis_batch_provider
common_agent_make_research_concept_hypothesis_batch_provider();

// Adapts host-verified contrasts to the existing grounding/TeachingRelation
// pipeline.  The callback is the host's dataset/tool/behavior resolver and
// must return only contrasts that it has already verified.
common_agent_concept_grounding_provider
common_agent_make_concept_grounding_provider(
        common_agent_concept_contrast_provider contrast_provider);
