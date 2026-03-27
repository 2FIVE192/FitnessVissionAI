package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.ActivityRepository
import com.app.fitness.DailyActivity
import com.app.fitness.User
import com.app.fitness.UserRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class HomeState(
    val user: User? = null,
    val activity: DailyActivity? = null,
    val isLoading: Boolean = true,
    val error: String? = null,
    // manual activity input
    val isManualSheetVisible: Boolean = false,
    val isSubmittingManual: Boolean = false,
    val manualSteps: String = "",
    val manualCalories: String = ""
)

class HomeViewModel(
    private val activityRepo: ActivityRepository,
    private val userRepo: UserRepository
) : ViewModel() {

    private val _state = MutableStateFlow(HomeState())
    val state = _state.asStateFlow()

    init {
        loadData()
    }

    fun showManualSheet() { _state.update { it.copy(isManualSheetVisible = true, manualSteps = "", manualCalories = "", error = null) } }
    fun hideManualSheet() { _state.update { it.copy(isManualSheetVisible = false) } }
    fun onManualStepsChanged(v: String) { _state.update { it.copy(manualSteps = v) } }
    fun onManualCaloriesChanged(v: String) { _state.update { it.copy(manualCalories = v) } }

    fun saveManualActivity() {
        val s = _state.value
        val steps = s.manualSteps.toIntOrNull() ?: 0
        val calories = s.manualCalories.toFloatOrNull() ?: return
        viewModelScope.launch {
            _state.update { it.copy(isSubmittingManual = true, error = null) }
            try {
                val updated = activityRepo.saveManualActivity(steps, calories)
                _state.update { it.copy(activity = updated, isManualSheetVisible = false, isSubmittingManual = false) }
            } catch (e: Exception) {
                _state.update { it.copy(isSubmittingManual = false, error = e.message) }
            }
        }
    }

    fun loadData() {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true) }

            try {
                val user = userRepo.getCurrentUser()
                val todayActivity = activityRepo.getTodayActivity()

                _state.update {
                    it.copy(
                        user = user,
                        activity = todayActivity,
                        isLoading = false,
                        error = null
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }
}

class HomeViewModelFactory(
    private val activityRepo: ActivityRepository,
    private val userRepo: UserRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return HomeViewModel(activityRepo, userRepo) as T
    }
}
