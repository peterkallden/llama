#pragma once

#include "agent/thinking/research/research-contract.h"
#include "agent/thinking/research/research-controller.h"

#include <functional>
#include <utility>


class common_agent_research_assessor {
public:
    virtual ~common_agent_research_assessor() = default;
    virtual common_agent_research_assessment assess(
            const common_agent_research_gap & gap,
            const common_agent_research_evidence & evidence) const = 0;
    virtual bool compare(const common_agent_research_gap &,
            const std::vector<common_agent_research_evidence> &,
            common_agent_research_source_comparison &) const { return false; }
};

// A host without a semantic assessor must not infer relevance from word overlap.
// The evidence remains available for a partial answer or a later assessment.
class common_agent_research_bounded_assessor final : public common_agent_research_assessor {
public:
    using callback_type = std::function<common_agent_research_assessment(
        const common_agent_research_gap &, const common_agent_research_evidence &)>;
    using comparison_callback_type = std::function<bool(const common_agent_research_gap &,
        const std::vector<common_agent_research_evidence> &, common_agent_research_source_comparison &)>;
    explicit common_agent_research_bounded_assessor(callback_type callback = {}, comparison_callback_type compare = {})
        : callback(std::move(callback)), comparison_callback(std::move(compare)) {}
    common_agent_research_assessment assess(
            const common_agent_research_gap & gap,
            const common_agent_research_evidence & evidence) const override {
        if (callback) return callback(gap, evidence);
        return {common_agent_research_assessment_status::inconclusive, 0.0,
            "Evidence has not been semantically assessed against the research question."};
    }
    bool compare(const common_agent_research_gap & gap,
            const std::vector<common_agent_research_evidence> & evidence,
            common_agent_research_source_comparison & result) const override {
        return comparison_callback && comparison_callback(gap, evidence, result);
    }
private:
    callback_type callback;
    comparison_callback_type comparison_callback;
};
