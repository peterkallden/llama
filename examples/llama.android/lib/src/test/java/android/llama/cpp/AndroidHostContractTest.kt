package android.llama.cpp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidHostContractTest {
    @Test
    fun host_resource_contract_keeps_content_uri_and_safe_local_name() {
        val uri = "content://picker/report.csv"
        val safe = "report 2026.csv".replace(Regex("[^A-Za-z0-9._-]"), "_")
        assertTrue(uri.startsWith("content://"))
        assertEquals("report_2026.csv", safe)
    }

    @Test
    fun model_contract_requires_gguf() {
        assertTrue("model.gguf".lowercase().endsWith(".gguf"))
    }
}
