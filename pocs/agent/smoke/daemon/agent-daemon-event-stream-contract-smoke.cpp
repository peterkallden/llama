#include "agent-daemon-events.h"
#include "agent-daemon-event-collector.h"
#include "agent-daemon-jsonl-protocol.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <memory>
#include <thread>
#include <utility>

int main() {
    const auto event = make_common_agent_daemon_event(
        common_agent_daemon_event_type::tool_completed,
        "request-1",
        "turn-1",
        "tool completed",
        7,
        {
            "namespace-1",
            "project-1",
            "session-1",
            "request-1",
            "turn-1",
            "operation-1",
        });

    common_agent_event_stream_subscription subscription;
    subscription.subscription_id = "subscription-1";
    subscription.filter.session_id = "session-1";
    subscription.filter.request_id = "request-1";
    subscription.cursor.after_sequence = 6;

    if (!subscription.filter.matches(event) || event.sequence <= subscription.cursor.after_sequence) {
        std::fprintf(stderr, "event stream contract did not match the expected event\n");
        return 1;
    }

    const auto failed_event = make_common_agent_daemon_event(
        common_agent_daemon_event_type::tool_failed,
        "request-1",
        "turn-1",
        "tool failed",
        8,
        {
            "namespace-1",
            "project-1",
            "session-1",
            "request-1",
            "turn-1",
            "operation-1",
        });
    if (failed_event.event_type != common_agent_daemon_event_type::tool_failed ||
            failed_event.category != common_agent_daemon_event_category::tool ||
            std::string(failed_event.type) != "tool.failed") {
        std::fprintf(stderr, "tool failure event contract was not preserved\n");
        return 1;
    }

    const common_agent_daemon_event_type tool_lifecycle[] = {
        common_agent_daemon_event_type::tool_queued,
        common_agent_daemon_event_type::tool_started,
        common_agent_daemon_event_type::tool_progress,
        common_agent_daemon_event_type::tool_output,
        common_agent_daemon_event_type::tool_artifact_created,
        common_agent_daemon_event_type::tool_completed,
        common_agent_daemon_event_type::tool_failed,
        common_agent_daemon_event_type::tool_cancelled,
        common_agent_daemon_event_type::tool_timed_out,
    };
    const char * tool_lifecycle_names[] = {
        "tool.queued",
        "tool.started",
        "tool.progress",
        "tool.output",
        "tool.artifact_created",
        "tool.completed",
        "tool.failed",
        "tool.cancelled",
        "tool.timed_out",
    };
    for (size_t i = 0; i < sizeof(tool_lifecycle) / sizeof(tool_lifecycle[0]); ++i) {
        const auto lifecycle_event = make_common_agent_daemon_event(
            tool_lifecycle[i], "request-1", "turn-1", {}, 0,
            {"namespace-1", "project-1", "session-1", "request-1", "turn-1", "operation-1"});
        if (std::string(lifecycle_event.type) != tool_lifecycle_names[i] ||
                lifecycle_event.operation_id != "operation-1") {
            std::fprintf(stderr, "tool lifecycle event contract was not preserved\n");
            return 1;
        }
    }

    const common_agent_daemon_event_type turn_and_inference_lifecycle[] = {
        common_agent_daemon_event_type::turn_started,
        common_agent_daemon_event_type::turn_resumed,
        common_agent_daemon_event_type::inference_queued,
        common_agent_daemon_event_type::inference_capacity_granted,
        common_agent_daemon_event_type::inference_started,
        common_agent_daemon_event_type::inference_completed,
    };
    const char * turn_and_inference_lifecycle_names[] = {
        "turn.started",
        "turn.resumed",
        "inference.queued",
        "inference.capacity_granted",
        "inference.started",
        "inference.completed",
    };
    for (size_t i = 0;
            i < sizeof(turn_and_inference_lifecycle) / sizeof(turn_and_inference_lifecycle[0]);
            ++i) {
        const auto lifecycle_event = make_common_agent_daemon_event(
            turn_and_inference_lifecycle[i], "request-1", "turn-1");
        const auto expected_category =
            turn_and_inference_lifecycle[i] == common_agent_daemon_event_type::inference_queued ||
            turn_and_inference_lifecycle[i] == common_agent_daemon_event_type::inference_capacity_granted ||
            turn_and_inference_lifecycle[i] == common_agent_daemon_event_type::inference_started ||
            turn_and_inference_lifecycle[i] == common_agent_daemon_event_type::inference_completed
                ? common_agent_daemon_event_category::inference
                : common_agent_daemon_event_category::turn;
        if (std::string(lifecycle_event.type) != turn_and_inference_lifecycle_names[i] ||
                lifecycle_event.category != expected_category) {
            std::fprintf(stderr, "turn/inference lifecycle event contract was not preserved\n");
            return 1;
        }
    }

    subscription.filter.turn_id = "other-turn";
    if (subscription.filter.matches(event)) {
        std::fprintf(stderr, "event stream contract ignored a turn filter\n");
        return 1;
    }

    const common_agent_event_stream_delivery delivery{
        common_agent_event_stream_delivery_kind::event,
        event,
        {event.sequence},
    };
    if (delivery.kind != common_agent_event_stream_delivery_kind::event ||
            delivery.cursor.after_sequence != event.sequence ||
            delivery.event.event_type != common_agent_daemon_event_type::tool_completed) {
        std::fprintf(stderr, "event stream delivery contract was not preserved\n");
        return 1;
    }

    common_agent_daemon_event_collector collector;
    common_agent_event_stream_subscription live_subscription;
    live_subscription.filter.session_id = "session-live";
    live_subscription.max_pending_events = 2;
    const auto live_id = collector.subscribe(live_subscription);

    collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::turn_started,
        "request-live",
        "turn-live",
        {},
        0,
        {"namespace-live", "project-live", "session-live", "request-live", "turn-live", {}}));
    common_agent_event_stream_delivery live_delivery;
    if (collector.wait_next(live_id, live_delivery, std::chrono::milliseconds(10)) !=
            common_agent_event_stream_wait_status::delivered ||
            live_delivery.event.event_type != common_agent_daemon_event_type::turn_started) {
        std::fprintf(stderr, "event stream collector did not deliver a matching event\n");
        return 1;
    }

    for (int i = 0; i < 3; ++i) {
        collector.append(make_common_agent_daemon_event(
            common_agent_daemon_event_type::tool_completed,
            "request-live",
            "turn-live",
            "queued",
            0,
            {"namespace-live", "project-live", "session-live", "request-live", "turn-live", {}}));
    }
    if (collector.wait_next(live_id, live_delivery, std::chrono::milliseconds(10)) !=
            common_agent_event_stream_wait_status::delivered ||
            live_delivery.kind != common_agent_event_stream_delivery_kind::overflow) {
        std::fprintf(stderr, "event stream collector did not report bounded overflow\n");
        return 1;
    }
    if (live_delivery.overflow_from_sequence == 0 ||
            live_delivery.overflow_to_sequence < live_delivery.overflow_from_sequence ||
            live_delivery.skipped_sequence_count == 0 ||
            live_delivery.skipped_sequence_count !=
                live_delivery.overflow_to_sequence - live_delivery.overflow_from_sequence + 1) {
        std::fprintf(stderr, "event stream collector did not report overflow metadata\n");
        return 1;
    }
    const auto live_overflow_delivery = live_delivery;

    collector.unsubscribe(live_id);
    if (collector.wait_next(live_id, live_delivery, std::chrono::milliseconds(10)) !=
            common_agent_event_stream_wait_status::closed) {
        std::fprintf(stderr, "event stream collector did not close a subscription\n");
        return 1;
    }

    common_agent_daemon_event_collector concurrent_collector;
    common_agent_event_stream_subscription concurrent_subscription;
    concurrent_subscription.filter.session_id = "session-concurrent";
    const auto concurrent_id = concurrent_collector.subscribe(concurrent_subscription);
    auto waiter_started = std::make_shared<std::promise<void>>();
    auto waiter_started_future = waiter_started->get_future();
    auto concurrent_waiter = std::async(
        std::launch::async,
        [&concurrent_collector, concurrent_id, waiter_started]() {
            common_agent_event_stream_delivery delivery;
            waiter_started->set_value();
            return std::make_pair(
                concurrent_collector.wait_next(
                    concurrent_id,
                    delivery,
                    std::chrono::seconds(2)),
                delivery);
        });
    if (waiter_started_future.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        std::fprintf(stderr, "event stream concurrent waiter did not start\n");
        return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < 2048; ++i) {
        common_agent_event_stream_subscription extra_subscription;
        extra_subscription.subscription_id = "concurrent-extra-" + std::to_string(i);
        concurrent_collector.subscribe(std::move(extra_subscription));
    }
    concurrent_collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::turn_started,
        "request-concurrent",
        "turn-concurrent",
        {},
        0,
        {"namespace-concurrent", "project-concurrent", "session-concurrent", "request-concurrent", "turn-concurrent", {}}));
    const auto concurrent_result = concurrent_waiter.get();
    if (concurrent_result.first != common_agent_event_stream_wait_status::delivered ||
            concurrent_result.second.event.event_type != common_agent_daemon_event_type::turn_started) {
        std::fprintf(stderr, "event stream concurrent subscribe/wait contract failed\n");
        return 1;
    }

    common_agent_daemon_event_collector replay_collector(8);
    common_agent_event_stream_subscription first_subscription;
    first_subscription.filter.session_id = "session-replay";
    const auto first_id = replay_collector.subscribe(first_subscription);
    replay_collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::turn_started,
        "request-replay",
        "turn-replay",
        "first",
        0,
        {"namespace-replay", "project-replay", "session-replay", "request-replay", "turn-replay", {}}));
    common_agent_event_stream_delivery first_delivery;
    if (replay_collector.wait_next(first_id, first_delivery, std::chrono::milliseconds(10)) !=
            common_agent_event_stream_wait_status::delivered) {
        std::fprintf(stderr, "event stream replay setup did not deliver the first event\n");
        return 1;
    }
    replay_collector.unsubscribe(first_id);
    replay_collector.append(make_common_agent_daemon_event(
        common_agent_daemon_event_type::turn_completed,
        "request-replay",
        "turn-replay",
        "second",
        0,
        {"namespace-replay", "project-replay", "session-replay", "request-replay", "turn-replay", {}}));
    common_agent_event_stream_subscription resumed_subscription;
    resumed_subscription.filter.session_id = "session-replay";
    resumed_subscription.cursor.after_sequence = first_delivery.cursor.after_sequence;
    const auto resumed_id = replay_collector.subscribe(resumed_subscription);
    common_agent_event_stream_delivery resumed_delivery;
    if (replay_collector.wait_next(resumed_id, resumed_delivery, std::chrono::milliseconds(10)) !=
            common_agent_event_stream_wait_status::delivered ||
            resumed_delivery.event.event_type != common_agent_daemon_event_type::turn_completed) {
        std::fprintf(stderr, "event stream replay did not resume from the cursor\n");
        return 1;
    }

    const auto jsonl_event = make_agent_daemon_jsonl_event_message("subscription-1", delivery);
    if (jsonl_event.value("message_type", "") != "event" ||
            jsonl_event.value("subscription_id", "") != "subscription-1" ||
            jsonl_event["event"].value("event_type", "") != "tool.completed" ||
            jsonl_event["event"].value("event_category", "") != "tool") {
        std::fprintf(stderr, "JSONL event projection did not preserve the contract\n");
        return 1;
    }
    const auto jsonl_failed_event = make_agent_daemon_jsonl_event_message(
        "subscription-1",
        {common_agent_event_stream_delivery_kind::event, failed_event, {failed_event.sequence}});
    if (jsonl_failed_event["event"].value("event_type", "") != "tool.failed") {
        std::fprintf(stderr, "JSONL tool failure projection did not preserve the contract\n");
        return 1;
    }
    const auto jsonl_overflow = make_agent_daemon_jsonl_event_message(
        "subscription-1", live_overflow_delivery);
    if (jsonl_overflow.value("delivery_kind", "") != "overflow" ||
            jsonl_overflow["overflow"].value("from_sequence", uint64_t(0)) !=
                live_overflow_delivery.overflow_from_sequence ||
            jsonl_overflow["overflow"].value("to_sequence", uint64_t(0)) !=
                live_overflow_delivery.overflow_to_sequence ||
            jsonl_overflow["overflow"].value("skipped_sequence_count", uint64_t(0)) !=
                live_overflow_delivery.skipped_sequence_count) {
        std::fprintf(stderr, "JSONL overflow projection did not preserve metadata\n");
        return 1;
    }

    std::printf("agent_event_stream_contract=ok\n");
    return 0;
}
