#pragma once

#include "agent-daemon-event-collector.h"
#include "agent-daemon-lifecycle.h"
#include "agent-daemon-events.h"
#include "../resource/agent-resource-store.h"
#include "../runtime/agent-runtime-session-manager.h"
#include "../runtime/agent-model-residency.h"
#include "../tooling/agent-tool-provider.h"

#include "memory/memory-store.h"
#include "agent/data-store.h"
#include "agent/turn-summary.h"
#include "plan/plan-store.h"

#include <memory>
#include <atomic>
#include <optional>
#include <string>
#include <vector>

class common_agent_inference_capacity_gate;

struct daemon_options;

class common_agent_daemon_config_store {
public:
    explicit common_agent_daemon_config_store(
        std::shared_ptr<const daemon_options> initial);

    std::shared_ptr<const daemon_options> snapshot() const;
    void replace(std::shared_ptr<const daemon_options> next);

private:
    mutable std::mutex mutex_;
    std::shared_ptr<const daemon_options> current_;
};

enum class common_agent_daemon_command_type {
    run_turn,
    execute_tool,
    cancel_turn,
    list_sessions,
    get_session,
    list_resources,
    list_memories,
    list_plans,
    reset_session,
    close_session,
    read_resource,
    put_resource,
    get_status,
    drain,
    shutdown,
    reload_config,
};

struct common_agent_daemon_reload_result {
    uint64_t config_version = 1;
    std::vector<std::string> applied_fields;
    std::vector<std::string> restart_required;
    std::vector<std::string> providers_added;
    std::vector<std::string> providers_removed;
    std::vector<std::string> providers_replaced;
    std::string warning;
};

struct common_agent_daemon_provider_readiness {
    std::string id;
    std::string status = "not_configured";
    bool required = false;
    std::string warning;
};

struct common_agent_daemon_runtime {
    std::shared_ptr<common_agent_daemon_config_store> config_store;
    std::unique_ptr<common_memory_store> memory_store;
    std::unique_ptr<common_plan_store> plan_store;
    std::unique_ptr<common_agent_data_store> data_store;
    std::unique_ptr<agent_resource_store> resource_store;
    std::shared_ptr<common_agent_runtime_model_residency> model_residency;
    common_agent_runtime_host_mode default_mode = common_agent_runtime_host_mode::chat;
    std::unique_ptr<common_agent_runtime_session_manager> host;
    std::shared_ptr<common_agent_inference_capacity_gate> inference_gate;
    common_agent_runtime_tooling provider_probe_tooling;
    std::vector<common_agent_daemon_provider_readiness> provider_readiness;
    std::function<bool(
        common_agent_runtime_tooling & tooling,
        std::vector<common_agent_daemon_provider_readiness> & providers,
        std::string & error)> probe_mcp_providers;
    std::function<bool(
        const struct common_agent_daemon_tool_payload & payload,
        agent_tool_result & result,
        std::string & error)> tool_executor;
    std::function<bool(
        const std::string & path,
        common_agent_daemon_reload_result & result,
        std::string & error)> reload_config;
};

struct common_agent_daemon_turn_payload {
    common_agent_runtime_session_manager_turn_request request;
    bool include_summary = false;
};

struct common_agent_daemon_tool_payload {
    common_agent_runtime_session_key session;
    std::string project_id;
    std::string tool_profile;
    std::string tool_name;
    std::string arguments_json = "{}";
    // Set by an authenticated caller when its write authority is narrower
    // than the daemon's configured profile. Empty preserves native defaults.
    std::optional<bool> allow_policy_gated_writes;
};

struct common_agent_daemon_session_payload {
    common_agent_runtime_session_key key;
};

struct common_agent_daemon_cancel_payload {
    std::string target_request_id;
    std::string target_turn_id;
};

struct common_agent_daemon_resource_payload {
    std::string uri;
    agent_resource_read_authority authority;
    size_t max_bytes = 8192;
};

struct common_agent_daemon_resource_put_payload {
    agent_resource_put_request request;
};

struct common_agent_daemon_scope_payload {
    agent_resource_read_authority authority;
};

struct common_agent_daemon_command {
    std::string request_id;
    common_agent_daemon_command_type type = common_agent_daemon_command_type::run_turn;
    std::optional<common_agent_daemon_turn_payload> turn;
    std::optional<common_agent_daemon_tool_payload> tool;
    std::optional<common_agent_daemon_session_payload> session;
    std::optional<common_agent_daemon_cancel_payload> cancel;
    std::optional<common_agent_daemon_resource_payload> resource;
    std::optional<common_agent_daemon_resource_put_payload> resource_put;
    std::optional<common_agent_daemon_scope_payload> scope;
    std::string reload_path;
};

struct common_agent_daemon_active_turn_status {
    std::string request_id;
    std::string turn_id;
    std::string phase;
    std::string disposition;
    bool cancellation_requested = false;
    std::string pending_operation_kind;
    std::string pending_operation_detail;
};

struct common_agent_daemon_tool_status {
    std::string name;
    std::string description;
    std::string source;
    std::string state = "active";
};

struct common_agent_daemon_readiness {
    // Lifecycle state and operational readiness are intentionally separate.
    // `health` describes whether the daemon can serve work now.
    std::string health = "failed";
    std::string model = "unavailable";
    std::string inference = "unavailable";
    std::string memory_store = "unavailable";
    std::string plan_store = "unavailable";
    std::string resource_store = "unavailable";
    std::string tool_profile;
    std::vector<common_agent_daemon_provider_readiness> providers;
    std::vector<common_agent_daemon_tool_status> tools;
    std::vector<std::string> warnings;
};

struct common_agent_daemon_status {
    common_agent_daemon_state state = common_agent_daemon_state::starting;
    bool live = false;
    bool ready = false;
    common_agent_daemon_readiness readiness;
    bool worker_running = false;
    size_t worker_count = 1;
    size_t workers_running = 0;
    bool accepting_commands = false;
    bool shutdown_requested = false;
    size_t queued_command_count = 0;
    size_t max_queue_size = 0;
    size_t queue_capacity_remaining = 0;
    size_t session_count = 0;
    std::optional<common_agent_daemon_active_turn_status> active_turn;
    std::string active_request_id;
    std::string active_turn_id;
    std::string active_turn_phase;
    std::string active_turn_disposition;
    bool active_cancel_requested = false;
    std::string active_pending_operation_kind;
    std::string active_pending_operation_detail;
    std::vector<common_agent_runtime_session_descriptor> sessions;
    bool session_snapshot_populated = false;
    uint64_t commands_accepted = 0;
    uint64_t commands_completed = 0;
    uint64_t commands_failed = 0;
    uint64_t turns_completed = 0;
    uint64_t tools_completed = 0;
    uint64_t config_version = 1;
};

enum class common_agent_daemon_response_kind {
    turn,
    tool,
    status,
    lifecycle,
    resource,
    listing,
};

struct common_agent_daemon_resource_read_result {
    agent_resource_descriptor resource;
    std::string content;
};

struct common_agent_daemon_memory_summary {
    std::string id;
    std::string kind;
    std::string scope;
    std::string summary;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
    int64_t created_at = 0;
};

struct common_agent_daemon_plan_summary {
    std::string plan_id;
    std::string purpose;
    std::string goal;
    std::string status;
    std::string scope;
    std::string session_id;
    std::string project_id;
    std::string turn_id;
    std::string active_step_id;
    std::string next_action;
    uint64_t version = 0;
    size_t step_count = 0;
    size_t observation_count = 0;
};

struct common_agent_daemon_listing_result {
    std::vector<agent_resource_descriptor> resources;
    std::vector<common_agent_daemon_memory_summary> memories;
    std::vector<common_agent_daemon_plan_summary> plans;
};

struct common_agent_daemon_command_outcome {
    bool ok = false;
    std::string request_id;
    common_agent_daemon_response_kind response_kind = common_agent_daemon_response_kind::turn;
    std::string event;
    std::string target_request_id;
    std::string target_turn_id;
    common_agent_daemon_status status;
    common_agent_runtime_session_manager_turn_result turn_result;
    agent_tool_result tool_result;
    common_agent_daemon_resource_read_result resource_result;
    common_agent_daemon_listing_result listing_result;
    common_agent_daemon_reload_result reload_result;
    std::optional<common_agent_turn_summary> turn_summary;
    std::string error;
};

struct common_agent_daemon_command_execution {
    common_agent_daemon_command_outcome outcome;
    std::vector<common_agent_daemon_event> events;
};

struct common_agent_daemon_command_result : common_agent_daemon_command_outcome {
    size_t daemon_event_count = 0;
    std::vector<common_agent_daemon_event> events;
};

common_agent_daemon_command_result project_agent_daemon_command_execution(
    common_agent_daemon_command_execution execution);

void append_agent_daemon_execution_event(
    common_agent_daemon_command_execution & execution,
    common_agent_daemon_event event);

class common_agent_daemon_service {
public:
    explicit common_agent_daemon_service(
        common_agent_daemon_runtime runtime,
        std::unique_ptr<common_agent_daemon_event_collector> event_collector = nullptr);

    bool execute_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error);

    bool execute(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_result & result,
        std::string & error);

    bool shutdown_requested() const { return shutdown_requested_flag.load(); }
    common_agent_runtime_host_mode default_mode() const { return runtime.default_mode; }
    common_agent_daemon_state state() const { return state_value.load(); }
    common_agent_daemon_readiness readiness() const;
    common_agent_daemon_shutdown_mode shutdown_mode() const { return shutdown_mode_value.load(); }
    std::vector<common_agent_runtime_session_descriptor> list_sessions() const;

    void mark_stopping();
    void mark_stopped();

    bool populate_status_outcome(
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error) const;

    bool populate_status(
        common_agent_daemon_command_result & result,
        std::string & error) const;

    std::optional<common_agent_runtime_active_turn_descriptor> describe_active_turn() const;

    bool request_cancel_active_turn(
        const std::string & target_request_id,
        const std::string & target_turn_id,
        common_agent_runtime_active_turn_descriptor & active_turn,
        std::string & error);

    void emit_internal_event(common_agent_daemon_event event);

    std::vector<common_agent_daemon_event> take_internal_events();

    std::string subscribe_events(common_agent_event_stream_subscription subscription);
    void unsubscribe_events(const std::string & subscription_id);
    common_agent_event_stream_wait_status wait_for_event(
        const std::string & subscription_id,
        common_agent_event_stream_delivery & delivery,
        std::chrono::milliseconds timeout);

private:
    void initialize_command_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const;

    void initialize_lifecycle_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const;

    void initialize_turn_outcome(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome) const;

    bool fail_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type) const;

    bool fail_turn_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type) const;

    bool succeed_lifecycle_result(
        const common_agent_daemon_command & command,
        common_agent_daemon_command_outcome & outcome,
        std::vector<common_agent_daemon_event> & events,
        std::string & error,
        std::string event,
        common_agent_daemon_event_type event_type,
        std::string detail = {}) const;

    common_agent_daemon_runtime runtime;
    std::unique_ptr<common_agent_daemon_event_collector> event_collector;
    std::atomic<common_agent_daemon_state> state_value = common_agent_daemon_state::starting;
    std::atomic<common_agent_daemon_shutdown_mode> shutdown_mode_value = common_agent_daemon_shutdown_mode::drain;
    std::atomic<bool> shutdown_requested_flag = false;
};
