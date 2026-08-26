#include "tools/agent/resource/agent-resource-store.h"
#include "tools/agent/resource/agent-resource-processing-service.h"
#include "tools/agent/resource/processors/agent-pdf-text-processor.h"
#include "tools/agent/tooling/agent-tool-runtime-adapter.h"
#include "tools/agent/tooling/agent-tool-provider.h"

#include "agent/tooling/catalog/tool-catalog.h"
#include "memory/memory-in-memory.h"

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

namespace {

agent_catalogued_resource_store g_resource_store(
    std::make_shared<agent_in_memory_blob_store>(),
    std::make_unique<agent_in_memory_resource_catalog>());

bool has_tool(const std::vector<common_chat_tool> & tools, const std::string & name) {
    for (const auto & tool : tools) {
        if (tool.name == name) {
            return true;
        }
    }
    return false;
}

bool has_name(const std::vector<std::string> & names, const std::string & name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

class recording_resource_processing_provider final
    : public agent_resource_processing_provider {
public:
    recording_resource_processing_provider(
            agent_resource_processing_provider & delegate,
            std::string & last_operation_id)
        : delegate_(delegate), last_operation_id_(last_operation_id) {}

    agent_resource_processing_result process(
            const agent_resource_processing_binding_request & request) const override {
        last_operation_id_ = request.operation_id;
        return delegate_.process(request);
    }

private:
    agent_resource_processing_provider & delegate_;
    std::string & last_operation_id_;
};

} // namespace

int main() {
    std::string error;
    common_tool_catalog catalog;
    common_tool_bootstrap_result bootstrap;
    if (!catalog.bootstrap("research", bootstrap, error)) {
        std::fprintf(stderr, "tool bootstrap failed: %s\n", error.c_str());
        return 1;
    }
    const auto all_configured = catalog.load_profile("all-configured", error);
    bool all_configured_has_data_join = false;
    bool all_configured_has_data_aggregate = false;
    bool all_configured_has_repository_log = false;
    bool all_configured_has_repository_status = false;
    bool all_configured_has_repository_changed_files = false;
    for (const auto & definition : all_configured) {
        all_configured_has_data_join = all_configured_has_data_join || definition.name == "data.join";
        all_configured_has_data_aggregate = all_configured_has_data_aggregate || definition.name == "data.aggregate";
        all_configured_has_repository_log = all_configured_has_repository_log || definition.name == "repository.log";
        all_configured_has_repository_status = all_configured_has_repository_status || definition.name == "repository.status";
        all_configured_has_repository_changed_files = all_configured_has_repository_changed_files || definition.name == "repository.changed_files";
    }
    if (!all_configured_has_data_join || !all_configured_has_data_aggregate ||
            !all_configured_has_repository_log || !all_configured_has_repository_status ||
            !all_configured_has_repository_changed_files) {
        std::fprintf(stderr, "all-configured profile did not expose the complete catalog required by this smoke\n");
        return 1;
    }

    native_agent_tool_provider minimal_provider(
        catalog,
        [](const agent_tool_context &, common_native_tool_bindings &, std::string &) {
            return true;
        });

    agent_tool_context minimal_context;
    minimal_context.request_id = "provider-smoke";
    minimal_context.turn_id = "turn-1";
    minimal_context.profile_id = "minimal";
    minimal_context.max_calls = 1;

    std::unique_ptr<agent_tool_view> minimal_view = minimal_provider.resolve_tools(minimal_context, error);
    if (!minimal_view) {
        std::fprintf(stderr, "minimal provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(minimal_view->chat_tools(), "math.calculate")) {
        std::fprintf(stderr, "calculator was not exposed through minimal tool view\n");
        return 1;
    }

    const auto first_result = minimal_view->call({
        "call-1",
        "math.calculate",
        R"({"expression":"17 * 23"})",
    }, error);
    if (!first_result.ok) {
        std::fprintf(stderr, "calculator call failed: %s\n", error.c_str());
        return 1;
    }
    if (first_result.content_json.find("391") == std::string::npos) {
        std::fprintf(stderr, "calculator result payload did not contain expected value: %s\n", first_result.content_json.c_str());
        return 1;
    }

    const auto second_result = minimal_view->call({
        "call-2",
        "math.calculate",
        R"({"expression":"1 + 1"})",
    }, error);
    if (second_result.ok ||
            second_result.failure_class != common_tool_failure_class::limit ||
            second_result.failure_code != "tool_call_limit_reached") {
        std::fprintf(stderr, "tool call limit was not enforced\n");
        return 1;
    }

    agent_tool_context cancelled_context = minimal_context;
    cancelled_context.execution_control = make_common_agent_runtime_execution_control({});
    cancelled_context.execution_control.cancellation->request_cancel("smoke cancelled");
    std::unique_ptr<agent_tool_view> cancelled_view = minimal_provider.resolve_tools(cancelled_context, error);
    if (!cancelled_view) {
        std::fprintf(stderr, "cancelled provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    const auto cancelled_result = cancelled_view->call({
        "call-cancelled",
        "math.calculate",
        R"({"expression":"2 + 2"})",
    }, error);
    if (cancelled_result.ok ||
            cancelled_result.failure_code != "tool_call_cancelled") {
        std::fprintf(stderr, "cancelled tool call did not return the expected failure\n");
        return 1;
    }

    agent_pdf_text_processor pdf_text_processor;
    agent_resource_processor_registry processor_registry;
    if (!processor_registry.add(pdf_text_processor, error)) {
        std::fprintf(stderr, "PDF text processor registration failed: %s\n", error.c_str());
        return 1;
    }
    agent_resource_processing_service processing_service(g_resource_store, processor_registry);
    std::string last_processing_operation_id;

    native_agent_tool_provider research_provider(
        catalog,
        [&](const agent_tool_context & context, common_native_tool_bindings & bindings, std::string &) {
            bindings.repository_root = context.repository_root;
            bindings.resource_runtime.store = &g_resource_store;
            bindings.resource_processing_service = &processing_service;
            bindings.resource_processing_provider_factory =
                [&](const agent_resource_processing_binding_request &) -> std::shared_ptr<agent_resource_processing_provider> {
                    return std::make_shared<recording_resource_processing_provider>(
                        processing_service, last_processing_operation_id);
                };
            bindings.resource_runtime.namespace_id = context.scope.namespace_id;
            bindings.resource_runtime.session_id = context.scope.session_id;
            bindings.resource_runtime.project_id = context.scope.project_id;
            bindings.resource_runtime.turn_id = context.scope.turn_id;
            const agent_resource_runtime runtime = bindings.resource_runtime;
            bindings.web_search = [runtime](const std::string &) {
                agent_resource_put_request request;
                request.name = "web-search-results.json";
                request.description = "Full web search result set for the current turn.";
                request.mime_type = "application/json";
                request.text = R"({"results":[{"title":"stub issue","url":"https://example.com/stub"}],"provider":"stub"})";
                request.scope = common_runtime_resource_scope::turn;
                request.source_provider = "native";
                request.source_tool = "web.search";
                request.metadata = {
                    "Preserve the full bounded web search candidate set outside the inline model context.",
                    "Stubbed search candidates for resident inference.",
                    "Use this resource when a later step needs the complete candidate list.",
                    "Provider smoke uses stubbed results.",
                    {"resident inference", "llama.cpp"},
                    {},
                };
                apply_agent_resource_runtime(runtime, request);

                agent_resource_descriptor descriptor;
                std::string error;
                if (!runtime.store->put_text(request, descriptor, error)) {
                    return common_tool_execution_result::failure(
                        "tool.web_search.resource_store_failed",
                        common_tool_failure_class::execution,
                        false,
                        "Provider smoke failed to write a resource.",
                        error);
                }

                common_runtime_resource_ref resource = descriptor;
                return common_tool_execution_result::success(
                    R"({"results":[{"title":"stub issue"}],"provider":"stub"})",
                    "Web search returned one stub candidate; the full result set was stored as a turn resource.",
                    {resource});
            };
            bindings.web_fetch = [runtime](const std::string &) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
                agent_resource_put_request request;
                request.name = "web-fetch-result.json";
                request.description = "Full bounded web fetch payload for the current turn.";
                request.mime_type = "application/json";
                request.text = R"({"url":"https://example.com/stub","final_url":"https://example.com/stub","status":200,"content_type":"text/html","title":"Stub Fetch Title","text":"Stub fetch body text that is intentionally long enough to live in the stored resource payload rather than only inline.","truncated":false})";
                request.scope = common_runtime_resource_scope::turn;
                request.source_provider = "native";
                request.source_tool = "web.fetch";
                request.metadata = {
                    "Preserve the full bounded web fetch result outside the inline model context.",
                    "Fetched bounded page text for \"Stub Fetch Title\".",
                    "Use this resource when a later step needs the full fetched text or metadata rather than the inline excerpt.",
                    "Provider smoke uses stubbed fetched text.",
                    {"https://example.com/stub", "Stub Fetch Title"},
                    {},
                };
                apply_agent_resource_runtime(runtime, request);

                agent_resource_descriptor descriptor;
                std::string error;
                if (!runtime.store->put_text(request, descriptor, error)) {
                    return common_tool_execution_result::failure(
                        "tool.web_fetch.resource_store_failed",
                        common_tool_failure_class::execution,
                        false,
                        "Provider smoke failed to write a fetched resource.",
                        error);
                }

                common_runtime_resource_ref resource = descriptor;
                return common_tool_execution_result::success(
                    R"({"url":"https://example.com/stub","final_url":"https://example.com/stub","status":200,"content_type":"text/html","title":"Stub Fetch Title","text_excerpt":"Stub fetch body text that is intentionally long enough to live in the stored resource payload rather than only inline.","text_length":112,"truncated":false})",
                    "Fetched bounded page text for \"Stub Fetch Title\"; the full payload was stored as a turn resource.",
                    {resource});
            };
            return true;
        });

    agent_tool_context research_context;
    research_context.request_id = "provider-smoke";
    research_context.turn_id = "turn-2";
    research_context.profile_id = "research";
    research_context.allow_network = true;
    research_context.max_calls = 16;
    research_context.async_exposed_tool_names = {"web.fetch"};
    research_context.scope.namespace_id = "provider-smoke";
    research_context.scope.session_id = "session-1";
    research_context.scope.project_id = "project-1";
    research_context.scope.turn_id = "turn-2";

    std::unique_ptr<agent_tool_view> research_view = research_provider.resolve_tools(research_context, error);
    if (!research_view) {
        std::fprintf(stderr, "research provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(research_view->chat_tools(), "web.search")) {
        std::fprintf(stderr, "web_search was not exposed in the research tool view\n");
        return 1;
    }
    if (research_view->supports_async_call("web.search")) {
        std::fprintf(stderr, "web_search should not have been marked async in this smoke\n");
        return 1;
    }

    agent_tool_context all_tools_context = research_context;
    all_tools_context.profile_id = "all-configured";
    all_tools_context.allow_network = true;
    all_tools_context.repository_root = std::filesystem::current_path().string();
    std::unique_ptr<agent_tool_view> all_tools_view = research_provider.resolve_tools(all_tools_context, error);
    if (!all_tools_view) {
        std::fprintf(stderr, "all-configured provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    auto tool_runtime = make_provider_agent_tool_runtime(*all_tools_view);
    std::string resolved_tool;
    std::vector<std::string> candidates;
    if (!tool_runtime->resolve_tool_name("join", resolved_tool, candidates) ||
            resolved_tool != "data.join" || !candidates.empty()) {
        std::fprintf(stderr, "unique fuzzy tool resolution did not select data.join\n");
        return 1;
    }
    agent_tool_context developer_context = all_tools_context;
    developer_context.profile_id = "developer-read";
    std::unique_ptr<agent_tool_view> developer_view = research_provider.resolve_tools(developer_context, error);
    if (!developer_view) {
        std::fprintf(stderr, "developer-read provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    auto developer_tool_runtime = make_provider_agent_tool_runtime(*developer_view);
    if (developer_tool_runtime->resolve_tool_name("search", resolved_tool, candidates) ||
            !has_name(candidates, "workspace.search") ||
            !has_name(candidates, "repository.search")) {
        std::fprintf(stderr, "ambiguous fuzzy tool resolution did not preserve search candidates (resolved=%s, count=%zu)\n",
            resolved_tool.c_str(), candidates.size());
        for (const auto & candidate : candidates) std::fprintf(stderr, "  candidate=%s\n", candidate.c_str());
        return 1;
    }
    if (!research_view->supports_async_call("web.fetch")) {
        std::fprintf(stderr, "web_fetch should have been marked async in this smoke\n");
        return 1;
    }

    const auto search_result = research_view->call({
        "call-2b",
        "web.search",
        R"({"query":"resident inference architecture in llama.cpp","limit":5})",
    }, error);
    if (!search_result.ok) {
        std::fprintf(stderr, "web_search call failed: %s\n", search_result.content_json.c_str());
        return 1;
    }
    if (search_result.resource_refs.empty()) {
        std::fprintf(stderr, "web_search did not materialize a resource reference for the full result set\n");
        return 1;
    }
    if (search_result.resource_refs[0].metadata.content_summary.empty()) {
        std::fprintf(stderr, "web_search resource metadata content summary was empty\n");
        return 1;
    }
    if (search_result.content_json.find("\"resources\"") == std::string::npos) {
        std::fprintf(stderr, "web_search result payload did not include rendered resources: %s\n", search_result.content_json.c_str());
        return 1;
    }

    agent_tool_pending_call pending_fetch;
    if (!research_view->begin_call_async({
            "call-2d",
            "web.fetch",
            R"({"url":"https://example.com/stub","max_bytes":64000,"extract":"text"})",
        }, pending_fetch, error)) {
        std::fprintf(stderr, "web_fetch async start failed: %s\n", error.c_str());
        return 1;
    }
    bool fetch_ready = false;
    agent_tool_result fetch_result;
    if (!research_view->poll_call_async(pending_fetch, fetch_ready, fetch_result, error)) {
        std::fprintf(stderr, "web_fetch async poll failed: %s\n", error.c_str());
        return 1;
    }
    if (fetch_ready) {
        std::fprintf(stderr, "web_fetch async operation completed too early for the smoke\n");
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    if (!research_view->poll_call_async(pending_fetch, fetch_ready, fetch_result, error)) {
        std::fprintf(stderr, "web_fetch async completion poll failed: %s\n", error.c_str());
        return 1;
    }
    if (!fetch_ready) {
        std::fprintf(stderr, "web_fetch async operation did not complete in time\n");
        return 1;
    }
    if (!fetch_result.ok) {
        std::fprintf(stderr, "web_fetch async call failed: %s\n", fetch_result.content_json.c_str());
        return 1;
    }

    agent_tool_pending_call cancelled_fetch;
    if (!research_view->begin_call_async({
            "call-2f",
            "web.fetch",
            R"({"url":"https://example.com/stub","max_bytes":64000,"extract":"text"})",
        }, cancelled_fetch, error)) {
        std::fprintf(stderr, "web_fetch cancellation async start failed: %s\n", error.c_str());
        return 1;
    }
    if (!research_view->cancel_call_async(cancelled_fetch, error)) {
        std::fprintf(stderr, "web_fetch cancellation signal failed: %s\n", error.c_str());
        return 1;
    }
    bool cancelled_fetch_ready = false;
    agent_tool_result cancelled_fetch_result;
    for (int attempt = 0; attempt < 40 && !cancelled_fetch_ready; ++attempt) {
        if (!research_view->poll_call_async(
                    cancelled_fetch,
                    cancelled_fetch_ready,
                    cancelled_fetch_result,
                    error)) {
            std::fprintf(stderr, "web_fetch cancellation poll failed: %s\n", error.c_str());
            return 1;
        }
        if (!cancelled_fetch_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    if (!cancelled_fetch_ready || cancelled_fetch_result.ok ||
            cancelled_fetch_result.failure_code != "tool_call_cancelled") {
        std::fprintf(stderr, "web_fetch cancellation did not produce the expected terminal result\n");
        return 1;
    }

    const auto sync_fetch_result = research_view->call({
        "call-2d",
        "web.fetch",
        R"({"url":"https://example.com/stub","max_bytes":64000,"extract":"text"})",
    }, error);
    if (!sync_fetch_result.ok) {
        std::fprintf(stderr, "web_fetch sync call failed: %s\n", sync_fetch_result.content_json.c_str());
        return 1;
    }
    if (fetch_result.resource_refs.empty()) {
        std::fprintf(stderr, "web_fetch did not materialize a resource reference for the full fetch payload\n");
        return 1;
    }
    if (fetch_result.resource_refs[0].metadata.content_summary.find("Stub Fetch Title") == std::string::npos) {
        std::fprintf(stderr, "web_fetch resource metadata content summary was unexpected\n");
        return 1;
    }
    if (fetch_result.content_json.find("\"text_excerpt\"") == std::string::npos ||
            fetch_result.content_json.find("\"resources\"") == std::string::npos) {
        std::fprintf(stderr, "web_fetch result payload did not include the expected inline excerpt and resources: %s\n", fetch_result.content_json.c_str());
        return 1;
    }

    const auto resource_read_result = research_view->call({
        "call-2c",
        "resource.read",
        std::string(R"({"uri":")") + search_result.resource_refs[0].uri + R"(","representation":"text","max_bytes":4096})",
    }, error);
    if (!resource_read_result.ok ||
            resource_read_result.content_json.find("stub issue") == std::string::npos ||
            resource_read_result.content_json.find("\"representation\":\"text\"") == std::string::npos) {
        std::fprintf(stderr, "resource_read did not return the expected stored payload: %s\n", resource_read_result.content_json.c_str());
        return 1;
    }

    const auto default_resource_read_result = research_view->call({
        "call-2d-default-text",
        "resource.read",
        std::string(R"({"uri":")") + search_result.resource_refs[0].uri + R"(","max_bytes":4096})",
    }, error);
    if (!default_resource_read_result.ok ||
            default_resource_read_result.content_json != resource_read_result.content_json) {
        std::fprintf(stderr, "resource_read did not default to the text representation: %s\n", default_resource_read_result.content_json.c_str());
        return 1;
    }

    const auto unavailable_representation_result = research_view->call({
        "call-2c-image",
        "resource.read",
        std::string(R"({"uri":")") + search_result.resource_refs[0].uri + R"(","representation":"image"})",
    }, error);
    if (unavailable_representation_result.ok ||
            unavailable_representation_result.failure_code != "tool.resource_read.representation_unavailable") {
        std::fprintf(stderr, "resource_read did not reject the unavailable representation: %s\n", unavailable_representation_result.content_json.c_str());
        return 1;
    }

    const auto fetch_resource_read_result = research_view->call({
        "call-2e",
        "resource.read",
        std::string(R"({"uri":")") + fetch_result.resource_refs[0].uri + R"(","max_bytes":4096})",
    }, error);
    if (!fetch_resource_read_result.ok || fetch_resource_read_result.content_json.find("Stub fetch body text") == std::string::npos) {
        std::fprintf(stderr, "resource_read did not return the expected stored fetch payload: %s\n", fetch_resource_read_result.content_json.c_str());
        return 1;
    }

    const auto resource_inspect_result = research_view->call({
        "call-2f",
        "resource.inspect",
        std::string(R"({"uri":")") + search_result.resource_refs[0].uri + R"("})",
    }, error);
    if (!resource_inspect_result.ok ||
            resource_inspect_result.content_json.find("available_representations") == std::string::npos ||
            resource_inspect_result.content_json.find("\"text\"") == std::string::npos) {
        std::fprintf(stderr, "resource_inspect did not return the expected representation metadata: %s\n", resource_inspect_result.content_json.c_str());
        return 1;
    }

    agent_resource_put_request pdf_request;
    pdf_request.name = "provider-report.pdf";
    pdf_request.description = "PDF source for host-owned resource processing coverage.";
    pdf_request.mime_type = "application/pdf";
    pdf_request.scope = common_runtime_resource_scope::turn;
    pdf_request.namespace_id = research_context.scope.namespace_id;
    pdf_request.session_id = research_context.scope.session_id;
    pdf_request.project_id = research_context.scope.project_id;
    pdf_request.turn_id = research_context.scope.turn_id;
    pdf_request.source_provider = "native";
    pdf_request.source_tool = "provider-smoke";
    pdf_request.bytes = "%PDF-1.7\n1 0 obj\n<< /Type /Page >>\nstream\nBT (First PDF line) Tj (Second PDF line) Tj ET\nendstream\nendobj\n";
    agent_resource_descriptor pdf_descriptor;
    if (!g_resource_store.put_bytes(pdf_request, pdf_descriptor, error)) {
        std::fprintf(stderr, "PDF resource setup failed: %s\n", error.c_str());
        return 1;
    }

    const auto processed_pdf_read = research_view->call({
        "call-2g-pdf-text",
        "resource.read",
        std::string(R"({"uri":")") + pdf_descriptor.uri + R"(","representation":"text","max_bytes":4096})",
    }, error);
    if (!processed_pdf_read.ok ||
            processed_pdf_read.content_json.find("First PDF line") == std::string::npos ||
            processed_pdf_read.content_json.find("Second PDF line") == std::string::npos ||
            processed_pdf_read.content_json.find("resource.process:pdf-text-local-v1") == std::string::npos) {
        std::fprintf(stderr, "resource_read did not materialize PDF text through the processing service: %s\n", processed_pdf_read.content_json.c_str());
        return 1;
    }
    if (last_processing_operation_id.rfind("resource-read/turn-2/", 0) != 0) {
        std::fprintf(stderr, "resource_read did not use the operation-scoped processing provider: %s\n", last_processing_operation_id.c_str());
        return 1;
    }

    const auto cached_pdf_read = research_view->call({
        "call-2g-pdf-text-cache",
        "resource.read",
        std::string(R"({"uri":")") + pdf_descriptor.uri + R"(","representation":"text","max_bytes":4096})",
    }, error);
    if (!cached_pdf_read.ok || cached_pdf_read.content_json != processed_pdf_read.content_json) {
        std::fprintf(stderr, "resource_read did not reuse the cached PDF text representation: %s\n", cached_pdf_read.content_json.c_str());
        return 1;
    }

    agent_resource_put_request binary_request;
    binary_request.name = "opaque-image.bin";
    binary_request.description = "Opaque binary resource for representation checks.";
    binary_request.mime_type = "application/octet-stream";
    binary_request.scope = common_runtime_resource_scope::turn;
    binary_request.namespace_id = research_context.scope.namespace_id;
    binary_request.session_id = research_context.scope.session_id;
    binary_request.project_id = research_context.scope.project_id;
    binary_request.turn_id = research_context.scope.turn_id;
    binary_request.source_provider = "native";
    binary_request.source_tool = "provider-smoke";
    binary_request.bytes = "PNG";
    binary_request.bytes.push_back('\0');
    binary_request.bytes.push_back(static_cast<char>(0xff));
    agent_resource_descriptor binary_descriptor;
    if (!g_resource_store.put_bytes(binary_request, binary_descriptor, error)) {
        std::fprintf(stderr, "binary resource setup failed: %s\n", error.c_str());
        return 1;
    }

    const auto binary_inspect_result = research_view->call({
        "call-2g",
        "resource.inspect",
        std::string(R"({"uri":")") + binary_descriptor.uri + R"("})",
    }, error);
    if (!binary_inspect_result.ok ||
            binary_inspect_result.content_json.find(R"("available_representations":["bytes"])" ) == std::string::npos) {
        std::fprintf(stderr, "resource_inspect exposed an unexpected binary representation: %s\n", binary_inspect_result.content_json.c_str());
        return 1;
    }

    const auto binary_read_result = research_view->call({
        "call-2h",
        "resource.read",
        std::string(R"({"uri":")") + binary_descriptor.uri + R"(","representation":"text"})",
    }, error);
    if (binary_read_result.ok ||
            binary_read_result.failure_code != "tool.resource_read.representation_unavailable") {
        std::fprintf(stderr, "resource_read did not reject binary text representation: %s\n", binary_read_result.content_json.c_str());
        return 1;
    }

    const auto binary_bytes_result = research_view->call({
        "call-2i",
        "resource.read",
        std::string(R"({"uri":")") + binary_descriptor.uri + R"(","representation":"bytes","max_bytes":32})",
    }, error);
    if (!binary_bytes_result.ok ||
            binary_bytes_result.content_json.find(R"("content_encoding":"base64")") == std::string::npos ||
            binary_bytes_result.content_json.find(R"("content":"UE5HAP8=")") == std::string::npos) {
        std::fprintf(stderr, "resource_read did not return bounded base64 bytes: %s\n", binary_bytes_result.content_json.c_str());
        return 1;
    }

    common_memory_in_memory_store memory_store;
    if (!memory_store.open("", error)) {
        std::fprintf(stderr, "memory store open failed: %s\n", error.c_str());
        return 1;
    }

    native_agent_tool_provider memory_provider(
        catalog,
        [&memory_store](const agent_tool_context &, common_native_tool_bindings & bindings, std::string &) {
            bindings.memory_store = &memory_store;
            bindings.memory_query.namespace_id = "provider-smoke";
            bindings.memory_query.session_id = "session-1";
            bindings.memory_query.project_id = "project-1";
            bindings.memory_query.scope = common_memory_scope::session;
            return true;
        });

    agent_tool_context memory_context;
    memory_context.request_id = "provider-smoke";
    memory_context.turn_id = "turn-3";
    memory_context.profile_id = "memory";
    memory_context.allow_policy_gated_writes = true;
    memory_context.allow_memory_proposals = true;

    std::unique_ptr<agent_tool_view> memory_view = memory_provider.resolve_tools(memory_context, error);
    if (!memory_view) {
        std::fprintf(stderr, "memory provider resolve failed: %s\n", error.c_str());
        return 1;
    }
    if (!has_tool(memory_view->chat_tools(), "memory.remember")) {
        std::fprintf(stderr, "memory_remember was not exposed through memory tool view\n");
        return 1;
    }

    const auto memory_result = memory_view->call({
        "call-3",
        "memory.remember",
        R"({"kind":"decision","content":"Use the existing memory store for symbolic project decisions before introducing a separate overlay system."})",
    }, error);
    if (!memory_result.ok || memory_result.content_json.find("\"decision\":\"accept\"") == std::string::npos) {
        std::fprintf(stderr, "memory_remember call did not succeed: %s\n", memory_result.content_json.c_str());
        return 1;
    }
    if (memory_result.content_json.find("\"kind\":\"decision\"") == std::string::npos) {
        std::fprintf(stderr, "memory_remember did not preserve the symbolic memory kind: %s\n", memory_result.content_json.c_str());
        return 1;
    }

    std::printf("provider_tools=%zu\n", minimal_view->chat_tools().size());
    std::printf("calculator_result=%s\n", first_result.content_json.c_str());
    std::printf("network_tool_exposed=%s\n", has_tool(research_view->chat_tools(), "web.search") ? "yes" : "no");
    std::printf("web_fetch_async=%s\n", fetch_ready ? "yes" : "no");
    std::printf("web_search_resource_uri=%s\n", search_result.resource_refs.empty() ? "" : search_result.resource_refs[0].uri.c_str());
    std::printf("web_fetch_resource_uri=%s\n", fetch_result.resource_refs.empty() ? "" : fetch_result.resource_refs[0].uri.c_str());
    std::printf("memory_remember_result=%s\n", memory_result.content_json.c_str());
    return 0;
}
