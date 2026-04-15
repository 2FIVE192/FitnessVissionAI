package com.app.fitness.mobile.viewmodel

import android.graphics.BitmapFactory
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate
import java.time.Period

data class FitVisionState(
    val user: User? = null,
    val history: List<DailyActivity> = emptyList(),
    val isLoading: Boolean = true,
    val error: String? = null,
    // photo
    val selectedPhotoBytes: ByteArray? = null,
    val selectedPhotoBitmap: ImageBitmap? = null,
    // generation options
    val activityType: String = "walking",        // walking | running | strength | cycling
    val periodMonths: Int = 6,                   // 1 | 3 | 6
    val goal: String = "lose_weight",            // lose_weight | gain_muscle | maintain
    // result
    val isGenerating: Boolean = false,
    val resultBitmap: ImageBitmap? = null,
    val generationMode: GenerationMode? = null
)

class FitVisionViewModel(
    private val generationRepo: GenerationRepository,
    private val userRepo: UserRepository,
    private val activityRepo: ActivityRepository
) : ViewModel() {

    private val _state = MutableStateFlow(FitVisionState())
    val state = _state.asStateFlow()

    init { loadData() }

    private fun loadData() {
        viewModelScope.launch {
            try {
                val user = userRepo.getCurrentUser()
                // Load 30 days of history for a representative activity average
                val history = activityRepo.getActivityHistory(30)
                _state.update { it.copy(user = user, history = history, isLoading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }

    fun onPhotoSelected(bytes: ByteArray) {
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
        _state.update {
            it.copy(
                selectedPhotoBytes = bytes,
                selectedPhotoBitmap = bitmap,
                resultBitmap = null,
                generationMode = null,
                error = null
            )
        }
    }

    fun onActivityTypeChanged(type: String) { _state.update { it.copy(activityType = type) } }
    fun onPeriodMonthsChanged(months: Int) { _state.update { it.copy(periodMonths = months) } }
    fun onGoalChanged(goal: String) { _state.update { it.copy(goal = goal) } }

    fun generate() {
        val s = _state.value
        val user = s.user ?: return
        val photoBytes = s.selectedPhotoBytes ?: return

        val age = Period.between(user.dateOfBirth, LocalDate.now()).years
        // Compute averages from saved history (only days with actual data)
        val activeDays = s.history.filter { it.steps > 0 || it.burnedCalories > 0 }
        val avgSteps = if (activeDays.isNotEmpty())
            (activeDays.sumOf { it.steps.toDouble() } / activeDays.size).toInt()
        else 0
        val avgCalories = if (activeDays.isNotEmpty())
            (activeDays.sumOf { it.burnedCalories } / activeDays.size).toFloat()
        else 0f

        viewModelScope.launch {
            _state.update { it.copy(isGenerating = true, error = null) }

            generationRepo.generate(
                GenerationRequest(
                    photoBytes = photoBytes,
                    gender = if (user.gender == Gender.MALE) "male" else "female",
                    age = age,
                    heightCm = user.heightCm.toInt(),
                    weightKg = user.weightKg,
                    avgStepsPerDay = avgSteps,
                    avgCaloriesPerDay = avgCalories,
                    activityType = s.activityType,
                    periodMonths = s.periodMonths,
                    goal = s.goal
                )
            ).onSuccess { result ->
                val bitmap = BitmapFactory
                    .decodeByteArray(result.imageBytes, 0, result.imageBytes.size)
                    ?.asImageBitmap()
                _state.update { it.copy(isGenerating = false, resultBitmap = bitmap, generationMode = result.mode) }
            }.onFailure { e ->
                _state.update { it.copy(isGenerating = false, error = e.message) }
            }
        }
    }
}

class FitVisionViewModelFactory(
    private val generationRepo: GenerationRepository,
    private val userRepo: UserRepository,
    private val activityRepo: ActivityRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return FitVisionViewModel(generationRepo, userRepo, activityRepo) as T
    }
}
