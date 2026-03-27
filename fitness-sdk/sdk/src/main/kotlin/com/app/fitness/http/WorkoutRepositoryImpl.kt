package com.app.fitness.http

import com.app.fitness.*
import com.app.fitness.http.DtoMapper.toDomain

/** HTTP implementation of [WorkoutRepository]. */
class WorkoutRepositoryImpl internal constructor(
    private val client: FitnessApiClient
) : WorkoutRepository {

    private val api get() = client.service

    override suspend fun getAllWorkouts(): List<Workout> {
        val resp = api.getAllWorkouts()
        check(resp.isSuccessful) { "getAllWorkouts failed: HTTP ${resp.code()}" }
        return (resp.body() ?: emptyList()).map { it.toDomain() }
    }

    override suspend fun getRecommendedWorkouts(): List<Workout> {
        val resp = api.getRecommendedWorkouts()
        check(resp.isSuccessful) { "getRecommendedWorkouts failed: HTTP ${resp.code()}" }
        return (resp.body() ?: emptyList()).map { it.toDomain() }
    }

    override suspend fun getAllFavoriteWorkouts(): List<Workout> {
        val resp = api.getFavoriteWorkouts()
        check(resp.isSuccessful) { "getAllFavoriteWorkouts failed: HTTP ${resp.code()}" }
        return (resp.body() ?: emptyList()).map { it.toDomain() }
    }

    override suspend fun getWorkoutById(id: String): Workout? {
        val resp = api.getWorkoutById(id)
        if (resp.code() == 404) return null
        check(resp.isSuccessful) { "getWorkoutById failed: HTTP ${resp.code()}" }
        return resp.body()?.toDomain()
    }

    override suspend fun toggleFavorite(workoutId: String): Result<Boolean> = runCatching {
        val resp = api.toggleFavorite(workoutId)
        check(resp.isSuccessful) { "toggleFavorite failed: HTTP ${resp.code()}" }
        checkNotNull(resp.body()).isFavorite
    }
}
