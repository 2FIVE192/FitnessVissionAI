package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.*
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class WorkoutDetailState(
    val workout: Workout? = null,
    val estimatedCalories: Double = 0.0,
    val sessionId: String? = null,
    val isSessionActive: Boolean = false,
    val elapsedSeconds: Int = 0,
    val currentExerciseIndex: Int = 0,
    val isLoading: Boolean = true,
    val error: String? = null,
    val completedMessage: String? = null,
    val isCompleted: Boolean = false
)

class WorkoutDetailViewModel(
    private val workoutId: String,
    private val workoutRepo: WorkoutRepository,
    private val sessionRepo: WorkoutSessionRepository,
    private val calculator: CalorieCalculatorService,
    private val userRepo: UserRepository
) : ViewModel() {

    private val _state = MutableStateFlow(WorkoutDetailState())
    val state = _state.asStateFlow()

    private var timerJob: Job? = null

    init {
        loadWorkout()
    }

    private fun loadWorkout() {
        viewModelScope.launch {
            try {
                val workout = workoutRepo.getWorkoutById(workoutId)
                val user = userRepo.getCurrentUser()

                // estimate how many calories this workout would burn
                val estimatedCals = if (workout != null) {
                    calculator.calculateWorkoutCalories(workout, user.weightKg)
                } else 0.0

                _state.update {
                    it.copy(
                        workout = workout,
                        estimatedCalories = estimatedCals,
                        isLoading = false
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }

    fun startSession() {
        viewModelScope.launch {
            try {
                val session = sessionRepo.startSession(workoutId)
                _state.update { it.copy(sessionId = session.id, isSessionActive = true, elapsedSeconds = 0, currentExerciseIndex = 0) }
                startTimer()
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (true) {
                delay(1000)
                val elapsed = _state.value.elapsedSeconds + 1
                val exercises = _state.value.workout?.exercises ?: emptyList()
                val totalSeconds = exercises.sumOf { it.durationSeconds + it.restAfterSeconds }
                _state.update { it.copy(elapsedSeconds = elapsed, currentExerciseIndex = computeExerciseIndex(elapsed, exercises)) }
                if (totalSeconds > 0 && elapsed >= totalSeconds) break
            }
            // all exercises finished — auto-complete the session
            val sid = _state.value.sessionId ?: return@launch
            try {
                val session = sessionRepo.completeSession(sid).getOrThrow()
                _state.update {
                    it.copy(isSessionActive = false, isCompleted = true, completedMessage = "Тренировка завершена! Сожжено ~${session.burnedCalories.toInt()} ккал")
                }
            } catch (e: Exception) {
                _state.update { it.copy(isSessionActive = false, error = e.message) }
            }
        }
    }

    private fun computeExerciseIndex(elapsedSeconds: Int, exercises: List<Exercise>): Int {
        var remaining = elapsedSeconds
        for ((index, exercise) in exercises.withIndex()) {
            remaining -= (exercise.durationSeconds + exercise.restAfterSeconds)
            if (remaining < 0) return index
        }
        return (exercises.size - 1).coerceAtLeast(0)
    }

    fun completeSession() {
        val sid = _state.value.sessionId ?: return
        timerJob?.cancel()

        viewModelScope.launch {
            try {
                val session = sessionRepo.completeSession(sid).getOrThrow()
                _state.update {
                    it.copy(
                        isSessionActive = false,
                        isCompleted = true,
                        completedMessage = "Тренировка завершена! Сожжено ~${session.burnedCalories.toInt()} ккал"
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun abandonSession() {
        val sid = _state.value.sessionId ?: return
        timerJob?.cancel()

        viewModelScope.launch {
            try {
                sessionRepo.abandonSession(sid)
                _state.update {
                    it.copy(isSessionActive = false, completedMessage = "Тренировка прервана")
                }
            } catch (e: Exception) {
                _state.update { it.copy(error = e.message) }
            }
        }
    }

    fun toggleFavorite() {
        viewModelScope.launch {
            runCatching { workoutRepo.toggleFavorite(workoutId) }
            loadWorkout()
        }
    }
}

class WorkoutDetailViewModelFactory(
    private val workoutId: String,
    private val workoutRepo: WorkoutRepository,
    private val sessionRepo: WorkoutSessionRepository,
    private val calculator: CalorieCalculatorService,
    private val userRepo: UserRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return WorkoutDetailViewModel(workoutId, workoutRepo, sessionRepo, calculator, userRepo) as T
    }
}
