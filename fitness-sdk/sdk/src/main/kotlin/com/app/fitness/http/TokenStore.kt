package com.app.fitness.http

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit

/**
 * Persists the JWT bearer token in a private [SharedPreferences] file.
 * Used internally by [FitnessApiClient]'s auth interceptor.
 */
internal class TokenStore(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences("fitness_sdk_prefs", Context.MODE_PRIVATE)

    var token: String?
        get() = prefs.getString(KEY_TOKEN, null)
        set(value) = prefs.edit { putString(KEY_TOKEN, value) }

    fun clear() = prefs.edit { remove(KEY_TOKEN) }

    companion object {
        private const val KEY_TOKEN = "jwt_token"
    }
}
