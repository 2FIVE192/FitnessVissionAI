package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.AuthRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class LoginState(
    val username: String = "",
    val password: String = "",
    val isLoading: Boolean = false,
    val error: String? = null,
    val isLoggedIn: Boolean = false
)

class LoginViewModel(private val authRepo: AuthRepository) : ViewModel() {

    private val _state = MutableStateFlow(LoginState())
    val state = _state.asStateFlow()

    fun onUsernameChanged(value: String) {
        _state.update { it.copy(username = value, error = null) }
    }

    fun onPasswordChanged(value: String) {
        _state.update { it.copy(password = value, error = null) }
    }

    fun login() {
        val current = _state.value
        val usernameEmpty = current.username.isBlank()
        val passwordEmpty = current.password.isBlank()

        if (usernameEmpty || passwordEmpty) {
            _state.update { it.copy(error = "заполните оба поля") }
            return
        }

        viewModelScope.launch {
            _state.update { it.copy(isLoading = true, error = null) }

            val result = authRepo.login(current.username, current.password)

            result.onSuccess {
                _state.update { it.copy(isLoading = false, isLoggedIn = true) }
            }.onFailure { err ->
                _state.update { it.copy(isLoading = false, error = err.message ?: "ошибка входа") }
            }
        }
    }
}

class LoginViewModelFactory(private val authRepo: AuthRepository) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return LoginViewModel(authRepo) as T
    }
}
