package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.ActivityRepository
import com.app.fitness.DailyActivity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

enum class ChartMetric { STEPS, CALORIES }

enum class HistoryRange(val days: Int, val label: String) {
    THREE_DAYS(3, "3 дня"),
    WEEK(7, "Неделя"),
    MONTH(30, "Месяц")
}

data class AnalyticsState(
    val history: List<DailyActivity> = emptyList(),
    val metric: ChartMetric = ChartMetric.STEPS,
    val range: HistoryRange = HistoryRange.WEEK,
    val isLoading: Boolean = true,
    val error: String? = null
)

class AnalyticsViewModel(
    private val activityRepo: ActivityRepository
) : ViewModel() {

    private val _state = MutableStateFlow(AnalyticsState())
    val state = _state.asStateFlow()

    init { loadHistory() }

    fun selectMetric(metric: ChartMetric) {
        _state.update { it.copy(metric = metric) }
    }

    fun selectRange(range: HistoryRange) {
        _state.update { it.copy(range = range) }
        loadHistory()
    }

    fun refresh() { loadHistory() }

    private fun loadHistory() {
        val days = _state.value.range.days
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true, error = null) }
            try {
                val history = activityRepo.getActivityHistory(days)
                _state.update { it.copy(history = history, isLoading = false) }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }
}

class AnalyticsViewModelFactory(
    private val activityRepo: ActivityRepository
) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return AnalyticsViewModel(activityRepo) as T
    }
}
