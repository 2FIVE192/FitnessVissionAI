package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.AuthRepository
import com.app.fitness.Gender
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.LocalDate

data class RegisterState(
    val name: String = "",
    val username: String = "",
    val password: String = "",
    val gender: Gender = Gender.MALE,
    val dateOfBirth: String = "2000-01-01",
    val weightKg: String = "70",
    val heightCm: String = "175",
    val stepsGoal: String = "10000",
    val caloriesGoal: String = "500",
    val isLoading: Boolean = false,
    val error: String? = null,
    val isRegistered: Boolean = false
)

class RegisterViewModel(private val authRepo: AuthRepository) : ViewModel() {

    private val _state = MutableStateFlow(RegisterState())
    val state = _state.asStateFlow()

    fun onNameChanged(v: String) { _state.update { it.copy(name = v, error = null) } }
    fun onUsernameChanged(v: String) { _state.update { it.copy(username = v, error = null) } }
    fun onPasswordChanged(v: String) { _state.update { it.copy(password = v, error = null) } }
    fun onGenderChanged(v: Gender) { _state.update { it.copy(gender = v) } }
    fun onDateOfBirthChanged(v: String) { _state.update { it.copy(dateOfBirth = v) } }
    fun onWeightChanged(v: String) { _state.update { it.copy(weightKg = v) } }
    fun onHeightChanged(v: String) { _state.update { it.copy(heightCm = v) } }
    fun onStepsGoalChanged(v: String) { _state.update { it.copy(stepsGoal = v) } }
    fun onCaloriesGoalChanged(v: String) { _state.update { it.copy(caloriesGoal = v) } }

    fun register() {
        val s = _state.value

        // basic validation
        val missingFields = s.name.isBlank() || s.username.isBlank() || s.password.length < 4
        if (missingFields) {
            _state.update { it.copy(error = "заполните все поля (пароль мин. 4 символа)") }
            return
        }

        val weight = s.weightKg.toFloatOrNull()
        val height = s.heightCm.toFloatOrNull()
        val steps = s.stepsGoal.toIntOrNull()
        val cals = s.caloriesGoal.toIntOrNull()
        val dob = runCatching { LocalDate.parse(s.dateOfBirth) }.getOrNull()

        val invalidNumbers = weight == null || height == null || steps == null || cals == null || dob == null
        if (invalidNumbers) {
            _state.update { it.copy(error = "проверьте числовые поля и формат даты (гггг-мм-дд)") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isLoading = true, error = null) }

            val result = authRepo.register(
                name = s.name,
                username = s.username,
                password = s.password,
                gender = s.gender,
                dateOfBirth = dob!!,
                weightKg = weight!!,
                heightCm = height!!,
                dailyStepsGoal = steps!!,
                dailyCaloriesGoal = cals!!
            )

            result.onSuccess {
                _state.update { it.copy(isLoading = false, isRegistered = true) }
            }.onFailure { err ->
                _state.update { it.copy(isLoading = false, error = err.message ?: "ошибка регистрации") }
            }
        }
    }
}

class RegisterViewModelFactory(private val authRepo: AuthRepository) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return RegisterViewModel(authRepo) as T
    }
}
