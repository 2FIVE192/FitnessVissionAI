package com.app.fitness.http

import com.app.fitness.CalorieCalculatorService
import com.app.fitness.Exercise
import com.app.fitness.Workout

/**
 * Pure-computation implementation of [CalorieCalculatorService].
 * No network calls — all calculations are performed locally.
 *
 * Formula: MET × weightKg × (durationSeconds / 3600.0)
 */
class CalorieCalculatorServiceImpl : CalorieCalculatorService {

    override fun calculateWorkoutCalories(workout: Workout, weightKg: Float): Double =
        workout.exercises.sumOf { ex -> calculateExerciseCalories(ex, weightKg) }

    private fun calculateExerciseCalories(exercise: Exercise, weightKg: Float): Double =
        exercise.met * weightKg * (exercise.durationSeconds / 3600.0)
}
