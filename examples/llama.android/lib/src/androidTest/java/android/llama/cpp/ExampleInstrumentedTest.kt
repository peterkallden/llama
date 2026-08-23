package android.llama.cpp

import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4

import org.junit.Test
import org.junit.runner.RunWith

import org.junit.Assert.*
import com.arm.aichat.AgentStorage
import java.io.File

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
        assertEquals("android.llama.cpp.test", appContext.packageName)
    }

    @Test
    fun nativeSqliteStorageRoundTrip() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.filesDir, "agent-storage-smoke")
        assertTrue(directory.mkdirs() || directory.isDirectory)
        assertTrue(AgentStorage.selfTest(directory.absolutePath))
    }
}
