package com.arm.aichat.agent

import android.content.Context
import android.net.Uri
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream

data class AndroidImportedResource(
    val uri: String,
    val path: String,
    val mimeType: String?,
    val sizeBytes: Long,
)

/** Imports ContentResolver data into the existing path-based agent resource seam. */
class AndroidResourceStore(private val context: Context) {
    fun import(uri: Uri, name: String = "resource-${System.currentTimeMillis()}"): AndroidImportedResource {
        val root = File(context.filesDir, "agent/resources").apply { mkdirs() }
        val target = File(root, safeName(name))
        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
            target.length()
        } ?: error("Unable to open Android resource URI: $uri")
        return AndroidImportedResource(uri.toString(), target.absolutePath,
            context.contentResolver.getType(uri), bytes)
    }

    fun delete(resource: AndroidImportedResource): Boolean =
        File(resource.path).takeIf { it.parentFile == File(context.filesDir, "agent/resources") }?.delete() == true

    /** Copies a local/content resource to a user-selected Android document. */
    fun copyTo(resourceUri: String, destination: Uri): Long {
        val input = openInput(resourceUri)
            ?: error("Unable to open local artifact resource: $resourceUri")
        return input.use { source ->
            openOutput(destination)?.use { output ->
                source.copyTo(output)
            } ?: error("Unable to open destination document: $destination")
            destinationLength(destination)
        }
    }

    private fun openInput(resourceUri: String) = when (val uri = Uri.parse(resourceUri)) {
        else -> when (uri.scheme) {
            "content", "file" -> context.contentResolver.openInputStream(uri)
            null -> FileInputStream(File(resourceUri))
            else -> error("Unsupported artifact URI scheme: ${uri.scheme}")
        }
    }

    private fun openOutput(destination: Uri) = when (destination.scheme) {
        "file" -> FileOutputStream(File(destination.path ?: error("Destination has no path")))
        else -> context.contentResolver.openOutputStream(destination)
    }

    private fun destinationLength(destination: Uri): Long = when (destination.scheme) {
        "file" -> File(destination.path ?: return 0L).length()
        else -> context.contentResolver.openAssetFileDescriptor(destination, "r")?.use { it.length }
            .takeIf { it != null && it >= 0 } ?: 0L
    }

    private fun safeName(value: String): String = value
        .replace(Regex("[^A-Za-z0-9._-]"), "_")
        .ifBlank { "resource" }
}
