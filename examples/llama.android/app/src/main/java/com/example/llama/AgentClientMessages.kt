package com.example.llama

import com.google.gson.JsonObject
import com.google.gson.JsonParser

data class AgentClientResult(
    val ok: Boolean,
    val response: String,
    val error: String,
)

data class AgentClientEvent(
    val type: String,
    val detail: String,
    val resourceUri: String,
    val json: String,
)

/** Small, testable decoder for the common terminal event/result envelopes. */
object AgentClientMessages {
    fun parseResult(message: String): AgentClientResult? = runCatching {
        val json = JsonParser.parseString(message).asJsonObject
        if (json.stringValue("message_type") != "response") return null
        val ok = json.booleanValue("ok")
        AgentClientResult(
            ok = ok,
            response = if (ok) json.stringValue("response") else "",
            error = if (ok) "" else json.stringValue("error").ifBlank { "Agent turn failed" },
        )
    }.getOrNull()

    fun eventDetail(message: String): String? = runCatching {
        val json = JsonParser.parseString(message).asJsonObject
        if (json.stringValue("message_type") != "event") return null
        json.objectValue("event")?.stringValue("detail")?.ifBlank { null }
    }.getOrNull()

    fun parseEvent(message: String): AgentClientEvent? = runCatching {
        val json = JsonParser.parseString(message).asJsonObject
        if (json.stringValue("message_type") != "event") return null
        val event = json.objectValue("event") ?: return null
        AgentClientEvent(
            type = event.stringValue("type").ifBlank { event.stringValue("event_type") },
            detail = event.stringValue("detail"),
            resourceUri = event.stringValue("resource_uri"),
            json = event.toString(),
        )
    }.getOrNull()

    private fun JsonObject.stringValue(name: String): String =
        get(name)?.takeUnless { it.isJsonNull }?.asString.orEmpty()

    private fun JsonObject.booleanValue(name: String): Boolean =
        get(name)?.takeUnless { it.isJsonNull }?.asBoolean ?: false

    private fun JsonObject.objectValue(name: String): JsonObject? =
        get(name)?.takeIf { it.isJsonObject }?.asJsonObject
}
