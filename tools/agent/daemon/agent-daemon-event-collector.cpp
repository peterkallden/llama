#include "agent-daemon-event-collector.h"

#include <algorithm>
#include <utility>

common_agent_daemon_event_collector::common_agent_daemon_event_collector(
        size_t max_history_events)
    : max_history_events(std::max<size_t>(1, max_history_events)) {}

void common_agent_daemon_event_collector::append(
        common_agent_daemon_event event) {
    std::lock_guard<std::mutex> lock(mutex);
    event.type = common_agent_daemon_event_type_name(event.event_type);
    event.category = common_agent_daemon_event_category_for_type(event.event_type);
    event.sequence = next_sequence++;
    pending_events.push_back(std::move(event));
    const auto & published = pending_events.back();
    history.push_back(published);
    while (history.size() > max_history_events) {
        history.pop_front();
    }
    for (auto & [subscription_id, state] : subscriptions) {
        (void) subscription_id;
        if (!state.active ||
                !state.subscription.filter.matches(published) ||
                published.sequence <= state.subscription.cursor.after_sequence) {
            continue;
        }
        if (state.pending.size() >= state.subscription.max_pending_events) {
            uint64_t overflow_from_sequence = published.sequence;
            if (!state.pending.empty()) {
                const auto & first_pending = state.pending.front();
                overflow_from_sequence = first_pending.kind ==
                        common_agent_event_stream_delivery_kind::event
                    ? first_pending.event.sequence
                    : (first_pending.overflow_from_sequence != 0
                        ? first_pending.overflow_from_sequence
                        : first_pending.cursor.after_sequence);
            }
            state.pending.clear();
            state.pending.push_back({
                common_agent_event_stream_delivery_kind::overflow,
                {},
                {published.sequence},
                overflow_from_sequence,
                published.sequence,
                published.sequence >= overflow_from_sequence
                    ? published.sequence - overflow_from_sequence + 1
                    : 0,
            });
            state.subscription.cursor.after_sequence = published.sequence;
            continue;
        }
        state.pending.push_back({
            common_agent_event_stream_delivery_kind::event,
            published,
            {published.sequence},
        });
        state.subscription.cursor.after_sequence = published.sequence;
    }
    condition.notify_all();
}

std::vector<common_agent_daemon_event> common_agent_daemon_event_collector::take() {
    std::lock_guard<std::mutex> lock(mutex);
    auto out = std::move(pending_events);
    pending_events.clear();
    return out;
}

std::string common_agent_daemon_event_collector::subscribe(
        common_agent_event_stream_subscription subscription) {
    std::lock_guard<std::mutex> lock(mutex);
    if (subscription.subscription_id.empty()) {
        subscription.subscription_id =
            "event-subscription-" + std::to_string(next_subscription_id++);
    }
    if (subscription.max_pending_events == 0) {
        subscription.max_pending_events = 1;
    }
    const std::string id = subscription.subscription_id;
    subscription_state state{std::move(subscription), {}, true};
    if (!history.empty()) {
        const uint64_t oldest_sequence = history.front().sequence;
        if (state.subscription.cursor.after_sequence + 1 < oldest_sequence) {
            state.pending.push_back({
                common_agent_event_stream_delivery_kind::overflow,
                {},
                {oldest_sequence - 1},
                state.subscription.cursor.after_sequence + 1,
                oldest_sequence - 1,
                oldest_sequence > state.subscription.cursor.after_sequence
                    ? oldest_sequence - state.subscription.cursor.after_sequence - 1
                    : 0,
            });
            state.subscription.cursor.after_sequence = oldest_sequence - 1;
        }
        for (const auto & event : history) {
            if (event.sequence <= state.subscription.cursor.after_sequence ||
                    !state.subscription.filter.matches(event)) {
                continue;
            }
            if (state.pending.size() >= state.subscription.max_pending_events) {
                const uint64_t overflow_from_sequence = state.pending.empty()
                    ? event.sequence
                    : (state.pending.front().kind ==
                            common_agent_event_stream_delivery_kind::event
                        ? state.pending.front().event.sequence
                        : state.pending.front().overflow_from_sequence);
                state.pending.clear();
                state.pending.push_back({
                    common_agent_event_stream_delivery_kind::overflow,
                    {},
                    {event.sequence},
                    overflow_from_sequence,
                    event.sequence,
                    event.sequence >= overflow_from_sequence
                        ? event.sequence - overflow_from_sequence + 1
                        : 0,
                });
                state.subscription.cursor.after_sequence = event.sequence;
                break;
            }
            state.pending.push_back({
                common_agent_event_stream_delivery_kind::event,
                event,
                {event.sequence},
            });
            state.subscription.cursor.after_sequence = event.sequence;
        }
    }
    subscriptions[id] = std::move(state);
    return id;
}

uint64_t common_agent_daemon_event_collector::latest_sequence() const {
    std::lock_guard<std::mutex> lock(mutex);
    return next_sequence == 0 ? 0 : next_sequence - 1;
}

void common_agent_daemon_event_collector::unsubscribe(
        const std::string & subscription_id) {
    std::lock_guard<std::mutex> lock(mutex);
    const auto it = subscriptions.find(subscription_id);
    if (it == subscriptions.end()) {
        return;
    }
    it->second.active = false;
    it->second.pending.clear();
    it->second.pending.push_back({
        common_agent_event_stream_delivery_kind::closed,
        {},
        it->second.subscription.cursor,
    });
    condition.notify_all();
}

common_agent_event_stream_wait_status common_agent_daemon_event_collector::wait_next(
        const std::string & subscription_id,
        common_agent_event_stream_delivery & delivery,
        std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    if (subscriptions.find(subscription_id) == subscriptions.end()) {
        return common_agent_event_stream_wait_status::not_found;
    }
    if (!condition.wait_for(lock, timeout, [this, &subscription_id]() {
            const auto current = subscriptions.find(subscription_id);
            return current == subscriptions.end() ||
                !current->second.pending.empty() || !current->second.active;
        })) {
        return common_agent_event_stream_wait_status::timeout;
    }
    const auto it = subscriptions.find(subscription_id);
    if (it == subscriptions.end()) {
        return common_agent_event_stream_wait_status::not_found;
    }
    if (it->second.pending.empty()) {
        return common_agent_event_stream_wait_status::closed;
    }
    delivery = std::move(it->second.pending.front());
    it->second.pending.pop_front();
    const auto status = delivery.kind == common_agent_event_stream_delivery_kind::closed
        ? common_agent_event_stream_wait_status::closed
        : common_agent_event_stream_wait_status::delivered;
    return status;
}
