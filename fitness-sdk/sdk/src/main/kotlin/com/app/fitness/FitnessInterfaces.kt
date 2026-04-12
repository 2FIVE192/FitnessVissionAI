package com.app.fitness

import kotlinx.coroutines.flow.Flow
import java.time.LocalDate
import java.time.LocalDateTime

// -- enums --

enum class Gender { MALE, FEMALE }

enum class WorkoutType { STRENGTH, CARDIO, STRETCHING, YOGA, HIIT }

enum class MuscleGroup { CHEST, BACK, ARMS, ABS, GLUTES, LEGS, FULL_BODY }

enum class DifficultyLevel { EASY, MEDIUM, HARD }

enum class DurationRange { SHORT, MEDIUM, LONG, EXTENDED }

// -- domain models --

data class User(
    val id: String,
    val name: String,
    val username: String,
    val gender: Gender,
    val dateOfBirth: LocalDate,
    val weightKg: Float,
    val heightCm: Float,
    val dailyStepsGoal: Int,
    val dailyCaloriesGoal: Int
)

data class Exercise(
    val id: String,
    val title: String,
    val muscleGroup: MuscleGroup,
    val met: Double,
    val durationSeconds: Int,
    val restAfterSeconds: Int = 0
)

data class Workout(
    val id: String,
    val title: String,
    val type: WorkoutType,
    val difficulty: DifficultyLevel,
    val durationMinutes: Int,
    val exercises: List<Exercise>,
    val isFavorite: Boolean = false
)

data class DailyActivity(
    val date: LocalDate,
    val steps: Int,
    val burnedCalories: Double,
    val distanceKm: Double
)

data class WorkoutSession(
    val id: String,
    val workoutId: String,
    val startedAt: LocalDateTime,
    val finishedAt: LocalDateTime?,
    val burnedCalories: Double,
    val completedEarly: Boolean
)

data class WorkoutFilter(
    val types: Set<WorkoutType> = emptySet(),
    val muscleGroups: Set<MuscleGroup> = emptySet(),
    val difficulties: Set<DifficultyLevel> = emptySet(),
    val durations: Set<DurationRange> = emptySet()
) {
    val isEmpty: Boolean
        get() = types.isEmpty() && muscleGroups.isEmpty() &&
                difficulties.isEmpty() && durations.isEmpty()
}

// -- auth --

interface AuthRepository {
    suspend fun isAuthenticated(): Boolean
    suspend fun login(username: String, password: String): Result<User>
    suspend fun register(
        name: String, username: String, password: String,
        gender: Gender, dateOfBirth: LocalDate,
        weightKg: Float, heightCm: Float,
        dailyStepsGoal: Int, dailyCaloriesGoal: Int
    ): Result<User>
    suspend fun logout()
}

// -- user / profile --

interface UserRepository {
    suspend fun getCurrentUser(): User
    suspend fun updateProfile(
        gender: Gender? = null, dateOfBirth: LocalDate? = null,
        weightKg: Float? = null, heightCm: Float? = null,
        username: String? = null, password: String? = null
    ): Result<User>
    suspend fun updateDailyGoals(stepsGoal: Int, caloriesGoal: Int): Result<User>
}

// -- activity --

interface ActivityRepository {
    suspend fun getTodayActivity(): DailyActivity
    suspend fun getActivityHistory(days: Int): List<DailyActivity>
    suspend fun saveSteps(totalStepsSinceBoot: Int, timestamp: LocalDateTime)
    suspend fun saveManualActivity(steps: Int, caloriesBurned: Float): DailyActivity
}

// -- workouts --

interface WorkoutRepository {
    suspend fun getAllWorkouts(): List<Workout>
    suspend fun getRecommendedWorkouts(): List<Workout>
    suspend fun getAllFavoriteWorkouts(): List<Workout>
    suspend fun getWorkoutById(id: String): Workout?
    suspend fun toggleFavorite(workoutId: String): Result<Boolean>
}

// -- workout session --

interface WorkoutSessionRepository {
    suspend fun startSession(workoutId: String): WorkoutSession
    suspend fun completeSession(
        sessionId: String,
        finishedAt: LocalDateTime = LocalDateTime.now()
    ): Result<WorkoutSession>
    suspend fun abandonSession(
        sessionId: String,
        finishedAt: LocalDateTime = LocalDateTime.now()
    ): Result<WorkoutSession>
}

// -- generation (FitVision AI face/body preview) --

enum class GenerationMode { GAIN_WEIGHT, MOTIVATE, REAL_PROGRESS }

data class GenerationRequest(
    val photoBytes: ByteArray,
    val gender: String,          // "male" | "female"
    val age: Int,
    val heightCm: Int,
    val weightKg: Float,
    val avgStepsPerDay: Int,
    val avgCaloriesPerDay: Float,
    val activityType: String,    // "running" | "walking" | "strength" | "cycling"
    val activeDaysPerWeek: Int,
    val periodMonths: Int,       // 1 | 3 | 6
    val goal: String             // "lose_weight" | "gain_muscle" | "maintain"
)

data class GenerationResult(
    val imageBytes: ByteArray,
    val mode: GenerationMode
)

interface GenerationRepository {
    suspend fun generate(request: GenerationRequest): Result<GenerationResult>
}

// -- calorie calculation (pure math, no network) --

interface CalorieCalculatorService {
    fun calculateWorkoutCalories(workout: Workout, weightKg: Float): Double
}

// -- step counter service (foreground service) --

interface StepCounterService {
    fun startTracking()
    fun stopTracking()
    val stepFlow: Flow<Int>
}

