package com.arm.aichat.agent

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.nio.ByteBuffer
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** Keystore-backed storage for host credentials; values never enter portable config. */
object AndroidCredentialStore {
    private const val KEY_ALIAS = "llama-agent-credentials"
    private const val PREFS_NAME = "llama-agent-credentials"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private var applicationContext: Context? = null

    @JvmStatic
    fun initialize(context: Context) {
        applicationContext = context.applicationContext
        key()
    }

    @JvmStatic
    fun put(reference: String, secret: String) {
        require(reference.isNotBlank()) { "credential reference must not be blank" }
        require(secret.isNotEmpty()) { "credential secret must not be empty" }
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val ciphertext = cipher.doFinal(secret.toByteArray(Charsets.UTF_8))
        val encrypted = ByteBuffer.allocate(4 + cipher.iv.size + ciphertext.size)
        encrypted.putInt(cipher.iv.size).put(cipher.iv).put(ciphertext)
        prefs().edit().putString(reference, Base64.encodeToString(encrypted.array(), Base64.NO_WRAP)).apply()
    }

    @JvmStatic
    fun resolve(reference: String): String? {
        if (reference.isBlank()) return null
        val encoded = prefs().getString(reference, null) ?: return null
        return try {
            val packed = ByteBuffer.wrap(Base64.decode(encoded, Base64.NO_WRAP))
            val ivSize = packed.int
            val iv = ByteArray(ivSize).also { packed.get(it) }
            val ciphertext = ByteArray(packed.remaining()).also { packed.get(it) }
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(128, iv))
            String(cipher.doFinal(ciphertext), Charsets.UTF_8)
        } catch (_: Exception) {
            null
        }
    }

    @JvmStatic
    fun remove(reference: String) {
        prefs().edit().remove(reference).apply()
    }

    private fun prefs() = requireNotNull(applicationContext) {
        "AndroidCredentialStore.initialize(context) must be called first"
    }.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private fun key(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val existing = keyStore.getKey(KEY_ALIAS, null) as? SecretKey
        if (existing != null) return existing
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(KeyGenParameterSpec.Builder(
            KEY_ALIAS,
            KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
        ).setBlockModes(KeyProperties.BLOCK_MODE_GCM)
            .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
            .build())
        return generator.generateKey()
    }
}
