package com.example.llama

import org.json.JSONObject

data class AgentClientResult(
    val ok: Boolean,
    val response: String,
    val error: String,
)

data class AgentClientEvent(
    val type: String,
    val detail: String,
    val json: String,
)

/** Small, testable decoder for the common terminal event/result envelopes. */
object AgentClientMessages {
    fun parseResult(message: String): AgentClientResult? = runCatching {
        val json = JSONObject(message)
        if (json.optString("message_type") != "response") return null
        val ok = json.optBoolean("ok", false)
        AgentClientResult(
            ok = ok,
            response = if (ok) json.optString("response") else "",
            error = if (ok) "" else json.optString("error").ifBlank { "Agent turn failed" },
        )
    }.getOrNull()

    fun eventDetail(message: String): String? = runCatching {
        val json = JSONObject(message)
        if (json.optString("message_type") != "event") return null
        json.optJSONObject("event")?.optString("detail")?.ifBlank { null }
    }.getOrNull()

    fun parseEvent(message: String): AgentClientEvent? = runCatching {
        val json = JSONObject(message)
        if (json.optString("message_type") != "event") return null
        val event = json.optJSONObject("event") ?: return null
        AgentClientEvent(
            type = event.optString("type").ifBlank { event.optString("event_type") },
            detail = event.optString("detail"),
            json = event.toString(2),
        )
    }.getOrNull()
}
