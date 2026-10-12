#pragma once

#include "agent/thinking/research/research-contract.h"
#include "agent/thinking/research/research-controller.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <unordered_set>

class common_agent_research_assessor {
public:
    virtual ~common_agent_research_assessor() = default;
    virtual common_agent_research_assessment assess(
            const common_agent_research_gap & gap,
            const common_agent_research_evidence & evidence) const = 0;
};

// Small deterministic assessor used until a host supplies a model-backed one.
// Generic criteria remain inconclusive so the existing bounded tool contract
// remains usable; explicit criteria require at least one meaningful token hit.
class common_agent_research_bounded_assessor final : public common_agent_research_assessor {
public:
    common_agent_research_assessment assess(
            const common_agent_research_gap & gap,
            const common_agent_research_evidence & evidence) const override {
        const auto criterion_tokens = meaningful_tokens(gap.completion_criterion);
        if (criterion_tokens.empty()) {
            return {common_agent_research_assessment_status::inconclusive,
                evidence.confidence, "completion criterion is generic; bounded evidence assessment remains inconclusive"};
        }
        const auto evidence_tokens = meaningful_tokens(evidence.statement);
        const bool matched = std::any_of(criterion_tokens.begin(), criterion_tokens.end(),
            [&](const std::string & token) {
                return evidence_tokens.find(token) != evidence_tokens.end();
            });
        return matched
            ? common_agent_research_assessment{
                common_agent_research_assessment_status::sufficient,
                std::min(1.0, std::max(0.0, evidence.confidence)),
                "evidence contains a bounded completion-criterion match"}
            : common_agent_research_assessment{
                common_agent_research_assessment_status::insufficient,
                std::min(0.5, std::max(0.0, evidence.confidence)),
                "evidence does not contain a bounded completion-criterion match"};
    }

private:
    static std::unordered_set<std::string> meaningful_tokens(const std::string & value) {
        static const std::unordered_set<std::string> stop_words = {
            "a", "an", "and", "are", "be", "directly", "evidence", "for", "must",
            "the", "to", "research", "gap", "criterion", "addresses", "address", "question"};
        std::string normalized;
        normalized.reserve(value.size());
        for (const unsigned char character : value) {
            normalized.push_back(std::isalnum(character) ? static_cast<char>(std::tolower(character)) : ' ');
        }
        std::istringstream stream(normalized);
        std::unordered_set<std::string> tokens;
        std::string token;
        while (stream >> token) {
            if (token.size() > 2 && !stop_words.count(token)) tokens.insert(token);
        }
        return tokens;
    }
};
