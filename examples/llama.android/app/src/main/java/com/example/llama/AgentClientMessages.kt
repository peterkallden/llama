package com.example.llama

import org.json.JSONObject

data class AgentClientResult(
    val ok: Boolean,
    val response: String,
    val error: String,
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
}
