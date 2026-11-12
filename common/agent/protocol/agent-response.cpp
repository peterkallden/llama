#include "agent-response.h"
#include <utility>
#include <set>

using json = nlohmann::ordered_json;

json common_agent_unverified_draft_json(const common_agent_unverified_draft & draft) {
    auto issues = json::array();
    for (const auto & item : draft.issues) {
        issues.push_back({{"id", item.id}, {"description", item.description},
            {"correction", item.correction}, {"evidence_ids", item.evidence_ids}});
    }
    return {{"text",draft.text},{"review_status","unverified"},{"stop_reason",draft.stop_reason},
        {"plan_id",draft.plan_id},{"draft_fingerprint",draft.draft_fingerprint},
        {"evidence_fingerprint",draft.evidence_fingerprint},{"issues",std::move(issues)}};
}

json common_agent_review_parts_json(const std::vector<common_agent_review_part> & parts) {
    auto out=json::array();
    for (const auto & part:parts) out.push_back({{"observation_id",part.observation_id},
        {"excerpt",part.excerpt},{"status",part.status},{"reason",part.reason},{"evidence_ids",part.evidence_ids},
        {"resource_uri",part.resource_uri},{"content_fingerprint",part.content_fingerprint},
        {"evidence_fingerprint",part.evidence_fingerprint}});
    return out;
}
bool common_agent_review_parts_from_json(const json & value,std::vector<common_agent_review_part> & out) {
    try {
        if(!value.is_array() || value.size()>8) return false;
        std::vector<common_agent_review_part> parts;
        std::set<std::pair<std::string,std::string>> targets;
        for(const auto & item:value) {
            common_agent_review_part part;
            part.observation_id=item.at("observation_id").get<std::string>();
            part.excerpt=item.at("excerpt").get<std::string>();
            part.status=item.at("status").get<std::string>();
            part.reason=item.at("reason").get<std::string>();
            part.evidence_ids=item.value("evidence_ids",std::vector<std::string>{});
            part.resource_uri=item.value("resource_uri",std::string{});
            part.content_fingerprint=item.value("content_fingerprint",std::string{});
            part.evidence_fingerprint=item.value("evidence_fingerprint",std::string{});
            if(part.observation_id.empty() || part.observation_id.size()>1024 || part.excerpt.empty() || part.excerpt.size()>512 ||
                part.reason.empty() || part.reason.size()>512 || part.evidence_ids.size()>8 ||
                (part.status!="reviewed" && part.status!="requires_work" && part.status!="unreviewed") ||
                part.resource_uri.size()>2048 || part.content_fingerprint.size()>128 || part.evidence_fingerprint.size()>128 ||
                !targets.emplace(part.observation_id,part.excerpt).second) return false;
            for(const auto & id:part.evidence_ids) if(id.empty() || id.size()>1024) return false;
            parts.push_back(std::move(part));
        }
        out=std::move(parts);return true;
    } catch(const json::exception &) { return false; }
}
json common_agent_review_progress_json(const common_agent_review_progress & progress) {
    auto remaining=json::array();
    for(const auto & item:progress.remaining) remaining.push_back({{"id",item.id},{"description",item.description},
        {"correction",item.correction},{"evidence_ids",item.evidence_ids}});
    json result = {{"schema","review-progress.v1"},{"plan_id",progress.plan_id},
        {"draft_fingerprint",progress.draft_fingerprint},{"evidence_fingerprint",progress.evidence_fingerprint},
        {"accepted",progress.accepted},{"paused",progress.paused},
        {"parts",common_agent_review_parts_json(progress.parts)},{"remaining",remaining},{"follow_up",progress.follow_up}};
    if (!progress.blocked_reason.empty()) result["blocked_reason"] = progress.blocked_reason;
    return result;
}
