package com.app.fitness

import android.content.Context
import com.app.fitness.http.*
import com.app.fitness.service.StepCounterServiceImpl

// central entry point for the fitness sdk.
// create one instance in your application class and pass it to viewmodels.
class FitnessSdk(
    private val context: Context,
    baseUrl: String,
    mlBaseUrl: String,
    enableLogging: Boolean = false
) {
    // shared api client used by all repositories
    private val apiClient = FitnessApiClient(
        context = context.applicationContext,
        baseUrl = baseUrl,
        enableLogging = enableLogging
    )

    // separate client for the ML backend (no auth, longer timeouts)
    private val fitVisionApiClient = FitVisionApiClient(mlBaseUrl)

    // -- public repositories --

    val auth: AuthRepository = AuthRepositoryImpl(apiClient)
    val user: UserRepository = UserRepositoryImpl(apiClient)
    val activity: ActivityRepository = ActivityRepositoryImpl(apiClient)
    val workouts: WorkoutRepository = WorkoutRepositoryImpl(apiClient)
    val sessions: WorkoutSessionRepository = WorkoutSessionRepositoryImpl(apiClient)
    val calorieCalculator: CalorieCalculatorService = CalorieCalculatorServiceImpl()
    val generation: GenerationRepository = GenerationRepositoryImpl(fitVisionApiClient)

    // -- step counter --

    @Volatile private var stepCountingStarted = false

    fun startStepCounting() {
        StepCounterServiceImpl.activityRepositoryRef = activity
        StepCounterServiceImpl.start(context.applicationContext)
        stepCountingStarted = true
    }

    fun stopStepCounting() {
        if (stepCountingStarted) {
            StepCounterServiceImpl.stop(context.applicationContext)
            stepCountingStarted = false
        }
    }

    // quick check if a token exists locally (no network)
    val hasStoredToken: Boolean get() = apiClient.tokenStore.token != null
}
