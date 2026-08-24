package com.arm.aichat.agent

import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URL
import javax.net.ssl.HttpsURLConnection

/** Platform HTTP/TLS seam for the native MCP client. Android owns trust validation. */
object AndroidMcpTransport {
    @JvmStatic
    fun post(
        urlValue: String,
        headersJson: String,
        body: String,
        connectTimeoutMs: Int,
        readTimeoutMs: Int,
        maxResultBytes: Int,
    ): String {
        require(urlValue.startsWith("https://")) { "Android MCP transport requires HTTPS" }
        val connection = URL(urlValue).openConnection() as? HttpsURLConnection
            ?: error("MCP endpoint did not create an HTTPS connection")
        try {
            connection.connectTimeout = connectTimeoutMs.coerceAtLeast(1)
            connection.readTimeout = readTimeoutMs.coerceAtLeast(1)
            connection.requestMethod = "POST"
            connection.doOutput = true
            val headersJsonObject = JSONObject(headersJson)
            headersJsonObject.keys().forEach { key ->
                connection.setRequestProperty(key, headersJsonObject.getString(key))
            }
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }
            val status = connection.responseCode
            val input = if (status >= 400) connection.errorStream else connection.inputStream
            val output = ByteArrayOutputStream()
            (input ?: ByteArrayInputStream(ByteArray(0))).use { stream ->
                val buffer = ByteArray(8192)
                while (true) {
                    val read = stream.read(buffer)
                    if (read < 0) break
                    if (output.size() + read > maxResultBytes) error("MCP HTTP response exceeded max_result_bytes")
                    output.write(buffer, 0, read)
                }
            }
            val headers = JSONObject()
            connection.headerFields.forEach { (key, values) ->
                if (key != null && !values.isNullOrEmpty()) headers.put(key, values.first())
            }
            return JSONObject()
                .put("status", status)
                .put("body", output.toString(Charsets.UTF_8.name()))
                .put("headers", headers)
                .toString()
        } finally {
            connection.disconnect()
        }
    }
}
