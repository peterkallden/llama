package com.example.llama

import android.app.Activity
import android.app.AlertDialog
import android.net.Uri
import android.os.Bundle
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

/** Hosts the existing agent web client, connected to an explicitly selected server. */
class AgentServerActivity : AppCompatActivity() {
    private lateinit var browser: WebView
    private var origin: Uri? = null
    private var pendingFiles: ValueCallback<Array<Uri>>? = null
    private val filePicker = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val files = if (result.resultCode == Activity.RESULT_OK) {
            WebChromeClient.FileChooserParams.parseResult(result.resultCode, result.data)
        } else null
        pendingFiles?.onReceiveValue(files)
        pendingFiles = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        browser = WebView(this)
        setContentView(browser)
        browser.settings.javaScriptEnabled = true
        browser.settings.domStorageEnabled = true
        browser.settings.allowFileAccess = false
        // File chooser results are Android content URIs; the WebView needs
        // access to the URI granted by the system picker for upload.
        browser.settings.allowContentAccess = true
        browser.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                val selected = origin ?: return true
                val url = request.url
                // The web application renders its first-party document apps in
                // sandboxed srcdoc frames. Keep the top-level server boundary
                // while allowing those empty frames to initialize.
                if (!request.isForMainFrame && url.scheme == "about" &&
                    (url.toString() == "about:blank" || url.toString() == "about:srcdoc")) return false
                return url.scheme != selected.scheme || url.host != selected.host || url.port != selected.port
            }
        }
        browser.webChromeClient = object : WebChromeClient() {
            override fun onShowFileChooser(
                webView: WebView, callback: ValueCallback<Array<Uri>>, params: FileChooserParams,
            ): Boolean {
                pendingFiles?.onReceiveValue(null)
                pendingFiles = callback
                return try {
                    filePicker.launch(params.createIntent())
                    true
                } catch (_: Exception) {
                    pendingFiles = null
                    callback.onReceiveValue(null)
                    false
                }
            }
        }
        val prefs = getSharedPreferences("agent-server", MODE_PRIVATE)
        val address = EditText(this).apply {
            hint = "https://agent.example.com/"
            setText(prefs.getString("url", ""))
            setSingleLine()
        }
        AlertDialog.Builder(this)
            .setTitle("Connect to agent server")
            .setMessage("Enter the HTTPS address of the agent web interface.")
            .setView(address)
            .setNegativeButton("Cancel") { _, _ -> finish() }
            .setOnCancelListener { finish() }
            .setPositiveButton("Connect") { _, _ ->
                val uri = Uri.parse(address.text.toString().trim())
                if (uri.scheme != "https" || uri.host.isNullOrBlank() || uri.userInfo != null) {
                    AlertDialog.Builder(this).setMessage("A valid HTTPS server address is required.")
                        .setPositiveButton("OK") { _, _ -> finish() }.show()
                } else {
                    origin = uri
                    prefs.edit().putString("url", uri.toString()).apply()
                    browser.loadUrl(uri.toString())
                }
            }.show()
    }

    override fun onDestroy() {
        pendingFiles?.onReceiveValue(null)
        pendingFiles = null
        browser.stopLoading()
        browser.destroy()
        super.onDestroy()
    }
}
