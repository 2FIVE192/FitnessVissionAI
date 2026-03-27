package com.app.fitness.mobile

import android.app.Application
import com.app.fitness.FitnessSdk

class FitnessApplication : Application() {

    // single sdk instance shared across the entire app
    lateinit var sdk: FitnessSdk
        private set

    override fun onCreate() {
        super.onCreate()

        // 10.0.2.2 is localhost from the Android emulator's perspective
        val backendUrl = "http://10.0.2.2:3000"
        val mlBackendUrl = "http://10.0.2.2:8000"

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
