package com.app.fitness.mobile

import android.app.Application
import com.app.fitness.FitnessSdk

class FitnessApplication : Application() {

    // single sdk instance shared across the entire app
    lateinit var sdk: FitnessSdk
        private set

    override fun onCreate() {
        super.onCreate()

        // URLs are injected at build time from local.properties (backend.url / ml.backend.url).
        // Defaults to 10.0.2.2 (host machine as seen from the Android emulator).
        val backendUrl   = BuildConfig.BACKEND_URL
        val mlBackendUrl = BuildConfig.ML_BACKEND_URL

        sdk = FitnessSdk(
            context = this,
            baseUrl = backendUrl,
            mlBaseUrl = mlBackendUrl,
            enableLogging = true
        )

        // restore pedometer state across app restarts
        val prefs = getSharedPreferences("fitness_prefs", MODE_PRIVATE)
        if (prefs.getBoolean("pedometer_enabled", false)) {
            sdk.startStepCounting()
        }
    }
}
