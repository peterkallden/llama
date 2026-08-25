package android.llama.cpp

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import com.arm.aichat.AgentStorage
import com.arm.aichat.agent.AndroidModelManager
import com.arm.aichat.agent.AndroidResourceStore
import android.net.Uri
import android.content.ComponentName
import java.io.File
import android.net.Uri
import com.arm.aichat.agent.AgentRuntimeService

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class ExampleInstrumentedTest {
    @Test
    fun useAppContext() {
        // Context of the app under test.
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        assertEquals("com.arm.aichat.test", appContext.packageName)
    }

    @Test
    fun nativeSqliteStorageRoundTrip() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "agent-storage-smoke")
        assertTrue(directory.mkdirs() || directory.isDirectory)
        assertTrue(AgentStorage.selfTest(directory.absolutePath))
    }

    @Test
    fun contentResolverResourceImportUsesPrivateAgentStorage() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(context.cacheDir, "resource-smoke.csv")
        source.writeText("name,value\norders,1\n")
        val imported = AndroidResourceStore(context).import(Uri.fromFile(source), "orders.csv")
        assertEquals("name,value\norders,1\n", File(imported.path).readText())
        assertTrue(imported.path.startsWith(context.filesDir.absolutePath))
        assertTrue(AndroidResourceStore(context).delete(imported))
        assertFalse(File(imported.path).exists())
    }

    @Test
    fun contentResolverModelImportKeepsGgufOutsideApk() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(context.cacheDir, "model-smoke.gguf")
        source.writeBytes(byteArrayOf(0x47, 0x47, 0x55, 0x46))
        val imported = AndroidModelManager(context).importGguf(Uri.fromFile(source))
        assertEquals(4, imported.sizeBytes)
        assertTrue(imported.path.startsWith(context.filesDir.absolutePath))
        assertTrue(AndroidModelManager(context).installed().any { it.path == imported.path })
    }

    @Test
    fun artifactExportCopiesBytesToUserSelectedDocument() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val source = File(context.cacheDir, "artifact-smoke.bin")
        val destination = File(context.cacheDir, "artifact-exported.bin")
        val payload = byteArrayOf(0, 1, 2, 3, 127, -1)
        source.writeBytes(payload)
        destination.delete()

        val copied = AndroidResourceStore(context).copyTo(
            source.absolutePath,
            Uri.fromFile(destination))

        assertEquals(payload.size.toLong(), copied)
        assertArrayEquals(payload, destination.readBytes())
        assertTrue(source.delete())
        assertTrue(destination.delete())
    }

    @Test
    fun agentRuntimeServiceIsHostOwnedAndNotExported() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val component = ComponentName(context, AgentRuntimeService::class.java)
        val info = context.packageManager.getServiceInfo(component, 0)

        assertFalse(info.exported)
        assertEquals(AgentRuntimeService.EXTRA_STORAGE_DIRECTORY,
            "com.arm.aichat.agent.STORAGE_DIRECTORY")
        assertEquals(AgentRuntimeService.EXTRA_MODEL_PATH,
            "com.arm.aichat.agent.MODEL_PATH")
    }
}
