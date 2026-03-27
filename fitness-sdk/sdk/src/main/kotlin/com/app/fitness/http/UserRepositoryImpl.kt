package com.app.fitness.http

import com.app.fitness.*
import com.app.fitness.http.DtoMapper.toDomain
import com.app.fitness.http.dto.UpdateGoalsRequest
import com.app.fitness.http.dto.UpdateProfileRequest
import java.time.LocalDate

/** HTTP implementation of [UserRepository]. */
class UserRepositoryImpl internal constructor(
    private val client: FitnessApiClient
) : UserRepository {

    private val api get() = client.service

    override suspend fun getCurrentUser(): User {
        val resp = api.getUser()
        check(resp.isSuccessful) { "getUser failed: HTTP ${resp.code()}" }
        return checkNotNull(resp.body()).toDomain()
    }

    override suspend fun updateProfile(
        gender: Gender?,
        dateOfBirth: LocalDate?,
        weightKg: Float?,
        heightCm: Float?,
        username: String?,
        password: String?
    ): Result<User> = runCatching {
        val resp = api.updateProfile(
            UpdateProfileRequest(
                gender = gender?.name,
                dateOfBirth = dateOfBirth?.toString(),
                weightKg = weightKg,
                heightCm = heightCm,
                username = username,
                password = password
            )
        )
        check(resp.isSuccessful) { "updateProfile failed: HTTP ${resp.code()}" }
        checkNotNull(resp.body()).toDomain()
    }

    override suspend fun updateDailyGoals(stepsGoal: Int, caloriesGoal: Int): Result<User> =
        runCatching {
            val resp = api.updateGoals(UpdateGoalsRequest(stepsGoal, caloriesGoal))
            check(resp.isSuccessful) { "updateGoals failed: HTTP ${resp.code()}" }
            checkNotNull(resp.body()).toDomain()
        }
}
