package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class ProfileState(
    val user: User? = null,
    val isLoading: Boolean = true,
    val isSaving: Boolean = false,
    val error: String? = null,
    val successMessage: String? = null,
    // editable fields
    val editWeightKg: String = "",
    val editHeightCm: String = "",
    val editStepsGoal: String = "",
    val editCaloriesGoal: String = ""
)

class ProfileViewModel(
    private val userRepo: UserRepository,
    private val authRepo: AuthRepository
) : ViewModel() {

    private val _state = MutableStateFlow(ProfileState())
    val state = _state.asStateFlow()

    init {
        loadProfile()
    }

    fun loadProfile() {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true) }

            try {
                val user = userRepo.getCurrentUser()

                _state.update {
                    it.copy(
                        user = user,
                        isLoading = false,
                        editWeightKg = user.weightKg.toString(),
                        editHeightCm = user.heightCm.toString(),
                        editStepsGoal = user.dailyStepsGoal.toString(),
                        editCaloriesGoal = user.dailyCaloriesGoal.toString()
                    )
                }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }

    fun onWeightChanged(v: String) { _state.update { it.copy(editWeightKg = v, error = null) } }
    fun onHeightChanged(v: String) { _state.update { it.copy(editHeightCm = v, error = null) } }
    fun onStepsGoalChanged(v: String) { _state.update { it.copy(editStepsGoal = v, error = null) } }
    fun onCaloriesGoalChanged(v: String) { _state.update { it.copy(editCaloriesGoal = v, error = null) } }

    fun saveProfile() {
        val s = _state.value
        val weight = s.editWeightKg.toFloatOrNull()
        val height = s.editHeightCm.toFloatOrNull()

        if (weight == null || height == null) {
            _state.update { it.copy(error = "введите корректные числовые значения") }
            return
        }
        if (weight < 20f || weight > 300f) {
            _state.update { it.copy(error = "вес: от 20 до 300 кг") }
            return
        }
        if (height < 100f || height > 250f) {
            _state.update { it.copy(error = "рост: от 100 до 250 см") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isSaving = true, error = null, successMessage = null) }

            userRepo.updateProfile(weightKg = weight, heightCm = height)
                .onSuccess { user ->
                    _state.update {
                        it.copy(
                            isSaving = false,
                            successMessage = "профиль обновлён",
                            user = user,
                            editWeightKg = user.weightKg.toString(),
                            editHeightCm = user.heightCm.toString()
                        )
                    }
                }
                .onFailure { e ->
                    _state.update { it.copy(isSaving = false, error = e.message) }
                }
        }
    }

    fun saveGoals() {
        val s = _state.value
        val steps = s.editStepsGoal.toIntOrNull()
        val cals = s.editCaloriesGoal.toIntOrNull()

        if (steps == null || cals == null) {
            _state.update { it.copy(error = "введите целые числа") }
            return
        }
        if (steps < 500 || steps > 50000) {
            _state.update { it.copy(error = "цель по шагам: от 500 до 50 000") }
            return
        }
        if (cals < 50 || cals > 5000) {
            _state.update { it.copy(error = "цель по калориям: от 50 до 5 000") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isSaving = true, error = null, successMessage = null) }

            userRepo.updateDailyGoals(steps, cals)
                .onSuccess { user ->
                    _state.update {
                        it.copy(
                            isSaving = false,
                            successMessage = "цели обновлены",
                            user = user,
                            editStepsGoal = user.dailyStepsGoal.toString(),
                            editCaloriesGoal = user.dailyCaloriesGoal.toString()
                        )
                    }
                }
                .onFailure { e ->
                    _state.update { it.copy(isSaving = false, error = e.message) }
                }
        }
    }

    fun logout() {
        viewModelScope.launch {
            authRepo.logout()
        }
    }
}

class ProfileViewModelFactory(
    private val userRepo: UserRepository,
    private val authRepo: AuthRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return ProfileViewModel(userRepo, authRepo) as T
    }
}
