#include "agent/tool-workflow-index.h"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>

namespace {

bool matches_family(const common_tool_workflow & workflow, const std::set<std::string> & selected) {
    return std::any_of(workflow.family_ids.begin(), workflow.family_ids.end(), [&](const std::string & family) {
        return selected.count(family) != 0;
    });
}

common_tool_workflow inspect_named() {
    return {
        "dataset.inspect_named",
        "Select a named dataset and inspect its structure.",
        {"dataset"},
        {"dataset.select", "dataset.inspect"},
        {"dataset.select(name=<name>) as=<dataset>", "dataset.inspect(dataset=$<dataset>.dataset)"},
        {"Create or select the dataset before consuming it.", "The alias is available only to later steps."}
    };
}

common_tool_workflow discover() {
    return {
        "dataset.discover",
        "List available datasets and select them by returned name.",
        {"dataset"},
        {"dataset.list", "dataset.select"},
        {"dataset.list() as=candidates", "dataset.select(name=<unique name from $candidates.names> or $candidates.names[index]) as=<dataset>"},
        {"Use an exact unique name from names or names[index] after dataset.list.", "Do not use $datasets[index] without a prior as:datasets."}
    };
}

common_tool_workflow join() {
    return {
        "dataset.join",
        "Select two datasets and join them on matching columns.",
        {"dataset", "data"},
        {"dataset.select", "data.join"},
        {"dataset.select(name=<left_name>) as=left", "dataset.select(name=<right_name>) as=right", "data.join(left=$left.dataset, right=$right.dataset, on=...) as=joined"},
        {"Join left and right remain explicit because they are semantic choices.", "Use $alias.dataset for an earlier selected result."}
    };
}

common_tool_workflow summarize() {
    return {
        "dataset.summarize",
        "Aggregate or describe a selected or produced dataset.",
        {"data", "statistics"},
        {"data.aggregate", "statistics.describe"},
        {"data.aggregate(dataset=$source.dataset, measures=..., group_by=...) as=aggregated", "statistics.describe(dataset=$source.dataset, columns=...)"},
        {"A dataset input may be omitted only when exactly one compatible source exists.", "Use an explicit alias when selecting among multiple datasets."}
    };
}

} // namespace

std::vector<common_tool_workflow> common_generate_tool_workflow_index() {
    return {discover(), inspect_named(), join(), summarize()};
}

std::vector<common_tool_workflow_slot> common_expand_tool_workflow_slots(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & workflow_ids) {
    (void) workflows;
    const auto selected = [&](const std::string & id) {
        return std::find(workflow_ids.begin(), workflow_ids.end(), id) != workflow_ids.end();
    };
    const bool want_discovery = selected("dataset.discover");
    const bool want_inspection = selected("dataset.inspect_named");
    const bool want_join = selected("dataset.join");
    const bool want_summary = selected("dataset.summarize");

    std::vector<common_tool_workflow_slot> slots;
    if (want_discovery) {
        slots.push_back({
            "discover", "dataset.list", "List the available registered datasets.", "candidates", {}});
    }

    if (want_join) {
        slots.push_back({
            "left_source", "dataset.select", "Select the left dataset by name or candidates index.", "left", {}});
        slots.push_back({
            "right_source", "dataset.select", "Select the right dataset by name or candidates index.", "right", {}});
    } else if (want_inspection || want_summary) {
        slots.push_back({
            "source", "dataset.select", "Select the dataset to inspect or summarize.", "source", {}});
    }

    if (want_inspection) {
        if (want_join) {
            slots.push_back({
                "inspect_left", "dataset.inspect", "Inspect the selected left dataset.", "", {{"dataset", "$left.dataset"}}});
            slots.push_back({
                "inspect_right", "dataset.inspect", "Inspect the selected right dataset.", "", {{"dataset", "$right.dataset"}}});
        } else {
            slots.push_back({
                "inspect_source", "dataset.inspect", "Inspect the selected dataset.", "", {{"dataset", "$source.dataset"}}});
        }
    }

    if (want_join) {
        slots.push_back({
            "join", "data.join", "Join the selected datasets on matching columns.", "joined",
                {{"left", "$left.dataset"}, {"right", "$right.dataset"}}});
    }

    if (want_summary) {
        const std::string source = want_join ? "$joined.dataset" : "$source.dataset";
        slots.push_back({
            "aggregate", "data.aggregate", "Aggregate values from the selected dataset.", "aggregated",
                {{"dataset", source}}});
        slots.push_back({
            "describe", "statistics.describe", "Describe the relevant numeric columns.", "",
                {{"dataset", source}}});
    }
    return slots;
}

std::string common_render_tool_workflow_index(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & family_ids,
        size_t max_chars) {
    const std::set<std::string> selected(family_ids.begin(), family_ids.end());
    std::string rendered = "workflows (choose a suitable composition before filling exact tool arguments):";
    for (const auto & workflow : workflows) {
        if (!matches_family(workflow, selected)) continue;
        std::ostringstream entry;
        entry << "\n- " << workflow.id << ": " << workflow.description << "\n  steps:";
        for (const auto & step : workflow.steps) entry << "\n    " << step;
        entry << "\n  rules:";
        for (const auto & rule : workflow.rules) entry << "\n    " << rule;
        const auto text = entry.str();
        if (rendered.size() + text.size() > max_chars) break;
        rendered += text;
    }
    return rendered;
}

std::string common_render_selected_tool_workflows(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & workflow_ids,
        size_t max_chars) {
    const std::set<std::string> selected(workflow_ids.begin(), workflow_ids.end());
    std::string rendered = "selected workflows:";
    for (const auto & workflow : workflows) {
        if (!selected.count(workflow.id)) continue;
        std::ostringstream entry;
        entry << "\n- " << workflow.id << ": " << workflow.description << "\n  steps:";
        for (const auto & step : workflow.steps) entry << "\n    " << step;
        entry << "\n  rules:";
        for (const auto & rule : workflow.rules) entry << "\n    " << rule;
        const auto text = entry.str();
        if (rendered.size() + text.size() > max_chars) break;
        rendered += text;
    }
    return rendered;
}

std::string common_tool_workflow_selection_schema() {
    return R"({"type":"object","additionalProperties":false,"required":["workflows"],"properties":{"workflows":{"type":"array","minItems":1,"maxItems":4,"uniqueItems":true,"items":{"type":"string","minLength":1,"maxLength":64}}}})";
}

bool common_parse_tool_workflow_selection(
        const std::string & json_text,
        common_tool_workflow_selection & selection,
        std::string & error) {
    selection = {};
    const auto value = nlohmann::ordered_json::parse(json_text, nullptr, false);
    if (value.is_discarded() || !value.is_object() || !value.contains("workflows") || !value["workflows"].is_array()) {
        error = "workflow selection requires workflows:string[]";
        return false;
    }
    if (value["workflows"].empty() || value["workflows"].size() > 4) {
        error = "workflow selection contains an invalid number of workflows";
        return false;
    }
    std::set<std::string> seen;
    for (const auto & item : value["workflows"]) {
        if (!item.is_string() || item.get<std::string>().empty() || !seen.insert(item.get<std::string>()).second) {
            error = "workflow selection workflows must contain unique non-empty strings";
            return false;
        }
        selection.workflow_ids.push_back(item.get<std::string>());
    }
    error.clear();
    return true;
}

bool common_validate_tool_workflow_plan(
        const std::vector<common_tool_workflow> & workflows,
        const std::vector<std::string> & workflow_ids,
        const std::vector<common_tool_workflow_step_view> & steps,
        std::string & error) {
    auto first_index = [&](const std::string & tool, size_t start = 0U) -> size_t {
        for (size_t i = start; i < steps.size(); ++i) {
            if (steps[i].tool_name == tool) return i;
        }
        return steps.size();
    };
    auto selected = [&](const std::string & id) {
        return std::find(workflow_ids.begin(), workflow_ids.end(), id) != workflow_ids.end();
    };

    if (selected("dataset.discover")) {
        const size_t list = first_index("dataset.list");
        const size_t select = first_index("dataset.select");
        if (list == steps.size() || select == steps.size() || list > select) {
            error = "workflow.required_producer_missing: dataset.discover requires dataset.list before dataset.select";
            return false;
        }
    }
    if (selected("dataset.inspect_named")) {
        const size_t select = first_index("dataset.select");
        const size_t inspect = first_index("dataset.inspect");
        if (select == steps.size() || inspect == steps.size() || select > inspect) {
            error = "workflow.required_producer_missing: dataset.inspect_named requires dataset.select before dataset.inspect";
            return false;
        }
    }
    if (selected("dataset.join")) {
        const size_t join = first_index("data.join");
        const size_t first_select = first_index("dataset.select");
        const size_t second_select = first_select == steps.size()
            ? steps.size() : first_index("dataset.select", first_select + 1);
        if (join == steps.size()) {
            error = "workflow.required_consumer_missing: dataset.join requires data.join";
            return false;
        }
        // Direct dataset identifiers can satisfy a join without select slots.
        // Model references require the two semantic source slots first.
        const bool has_model_reference = std::any_of(
            steps.begin(), steps.end(), [](const auto & step) {
                return step.arguments_json.find("\"$") != std::string::npos;
            });
        if (has_model_reference &&
                (first_select == steps.size() || second_select == steps.size() || second_select > join)) {
            error = "workflow.required_producer_missing: dataset.join requires two dataset.select steps before data.join, unless direct dataset references were supplied";
            return false;
        }
    }
    (void) workflows;
    error.clear();
    return true;
}
