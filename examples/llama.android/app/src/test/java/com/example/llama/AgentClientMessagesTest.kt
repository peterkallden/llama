package com.example.llama

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AgentClientMessagesTest {
    @Test
    fun parsesSuccessfulTerminalResult() {
        val result = AgentClientMessages.parseResult(
            "{\"message_type\":\"response\",\"request_id\":\"3\",\"ok\":true,\"response\":\"done\"}"
        )

        assertTrue(result?.ok == true)
        assertEquals("done", result?.response)
        assertEquals("", result?.error)
    }

    @Test
    fun parsesFailedTerminalResultWithoutResponseText() {
        val result = AgentClientMessages.parseResult(
            "{\"message_type\":\"response\",\"ok\":false,\"error\":\"cancelled\"}"
        )

        assertEquals(false, result?.ok)
        assertEquals("", result?.response)
        assertEquals("cancelled", result?.error)
    }

    @Test
    fun extractsEventDetailAndRejectsOtherMessages() {
        assertEquals("tool completed", AgentClientMessages.eventDetail(
            "{\"message_type\":\"event\",\"event\":{\"detail\":\"tool completed\"}}"
        ))
        assertNull(AgentClientMessages.parseResult("{\"message_type\":\"event\"}"))
        assertNull(AgentClientMessages.eventDetail("not-json"))
    }

    @Test
    fun parsesCompactEventAndRetainsExpandedJson() {
        val event = AgentClientMessages.parseEvent(
            "{\"message_type\":\"event\",\"event\":{\"type\":\"tool.completed\",\"detail\":\"done\",\"step_id\":\"step-1\"}}"
        )

        assertEquals("tool.completed", event?.type)
        assertEquals("done", event?.detail)
        assertTrue(event?.json?.contains("step-1") == true)
    }
}
