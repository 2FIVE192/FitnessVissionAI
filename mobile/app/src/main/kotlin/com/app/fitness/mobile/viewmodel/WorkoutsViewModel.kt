package com.app.fitness.mobile.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.app.fitness.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class WorkoutsState(
    val allWorkouts: List<Workout> = emptyList(),
    val recommendedWorkouts: List<Workout> = emptyList(),
    val favoriteWorkouts: List<Workout> = emptyList(),
    val filteredWorkouts: List<Workout> = emptyList(),
    val filter: WorkoutFilter = WorkoutFilter(),
    val isLoading: Boolean = true,
    val error: String? = null,
    val selectedTab: Int = 0 // 0=all, 1=recommended, 2=favorites
)

class WorkoutsViewModel(private val workoutRepo: WorkoutRepository) : ViewModel() {

    private val _state = MutableStateFlow(WorkoutsState())
    val state = _state.asStateFlow()

    init {
        loadWorkouts()
    }

    fun loadWorkouts() {
        viewModelScope.launch {
            _state.update { it.copy(isLoading = true) }

            try {
                val all = workoutRepo.getAllWorkouts()
                val recommended = workoutRepo.getRecommendedWorkouts()
                val favorites = runCatching { workoutRepo.getAllFavoriteWorkouts() }.getOrDefault(emptyList())

                _state.update {
                    val next = it.copy(
                        allWorkouts = all,
                        recommendedWorkouts = recommended,
                        favoriteWorkouts = favorites,
                        isLoading = false
                    )
                    next.copy(filteredWorkouts = computeFiltered(next))
                }
            } catch (e: Exception) {
                _state.update { it.copy(isLoading = false, error = e.message) }
            }
        }
    }

    fun selectTab(index: Int) {
        _state.update {
            val next = it.copy(selectedTab = index)
            next.copy(filteredWorkouts = computeFiltered(next))
        }
    }

    fun toggleFavorite(workoutId: String) {
        viewModelScope.launch {
            workoutRepo.toggleFavorite(workoutId).onSuccess { isFavorite ->
                _state.update { s ->
                    val update = { w: Workout -> if (w.id == workoutId) w.copy(isFavorite = isFavorite) else w }
                    val updatedAll = s.allWorkouts.map(update)
                    val updatedRecommended = s.recommendedWorkouts.map(update)
                    val updatedFavorites = if (isFavorite) {
                        val toAdd = updatedAll.find { it.id == workoutId }
                            ?: updatedRecommended.find { it.id == workoutId }
                        if (toAdd != null && s.favoriteWorkouts.none { it.id == workoutId })
                            s.favoriteWorkouts + toAdd
                        else s.favoriteWorkouts
                    } else {
                        s.favoriteWorkouts.filter { it.id != workoutId }
                    }
                    val next = s.copy(
                        allWorkouts = updatedAll,
                        recommendedWorkouts = updatedRecommended,
                        favoriteWorkouts = updatedFavorites
                    )
                    next.copy(filteredWorkouts = computeFiltered(next))
                }
            }.onFailure {
                _state.update { it.copy(error = "не удалось обновить избранное") }
            }
        }
    }

    fun applyFilter(filter: WorkoutFilter) {
        _state.update {
            val next = it.copy(filter = filter)
            next.copy(filteredWorkouts = computeFiltered(next))
        }
    }

    fun clearFilter() {
        _state.update {
            val next = it.copy(filter = WorkoutFilter())
            next.copy(filteredWorkouts = computeFiltered(next))
        }
    }

    private fun computeFiltered(state: WorkoutsState): List<Workout> {
        val base = when (state.selectedTab) {
            0 -> state.allWorkouts
            1 -> state.recommendedWorkouts
            2 -> state.favoriteWorkouts
            else -> state.allWorkouts
        }
        val f = state.filter
        if (f.isEmpty) return base
        return base.filter { w ->
            (f.types.isEmpty() || w.type in f.types) &&
            (f.difficulties.isEmpty() || w.difficulty in f.difficulties) &&
            (f.muscleGroups.isEmpty() || w.exercises.any { it.muscleGroup in f.muscleGroups }) &&
            (f.durations.isEmpty() || f.durations.any { dur ->
                val totalMin = w.exercises.sumOf { it.durationSeconds + it.restAfterSeconds } / 60.0
                when (dur) {
                    DurationRange.SHORT    -> totalMin < 3
                    DurationRange.MEDIUM   -> totalMin in 3.0..5.0
                    DurationRange.LONG     -> totalMin in 5.0..8.0
                    DurationRange.EXTENDED -> totalMin > 8
                }
            })
        }
    }
}

class WorkoutsViewModelFactory(private val workoutRepo: WorkoutRepository) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        @Suppress("UNCHECKED_CAST")
        return WorkoutsViewModel(workoutRepo) as T
    }
}
