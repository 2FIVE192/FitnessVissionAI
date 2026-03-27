package com.app.fitness.http

import com.app.fitness.*
import com.app.fitness.http.DtoMapper.toDomain
import com.app.fitness.http.dto.LoginRequest
import com.app.fitness.http.dto.RegisterRequest
import java.time.LocalDate

/**
 * HTTP implementation of [AuthRepository].
 * On successful login/register the JWT is persisted via [TokenStore] and
 * automatically attached to every subsequent request by the OkHttp interceptor.
 */
class AuthRepositoryImpl internal constructor(
    private val client: FitnessApiClient
) : AuthRepository {

    private val api get() = client.service
    private val tokenStore get() = client.tokenStore

    override suspend fun isAuthenticated(): Boolean = tokenStore.token != null

    override suspend fun login(username: String, password: String): Result<User> =
        runCatching {
            val resp = api.login(LoginRequest(username, password))
            check(resp.isSuccessful) { "Login failed: HTTP ${resp.code()}" }
            val body = checkNotNull(resp.body()) { "Empty response body" }
            tokenStore.token = body.token
            body.user.toDomain()
        }

    override suspend fun register(
        name: String,
        username: String,
        password: String,
        gender: Gender,
        dateOfBirth: LocalDate,
        weightKg: Float,
        heightCm: Float,
        dailyStepsGoal: Int,
        dailyCaloriesGoal: Int
    ): Result<User> = runCatching {
        val resp = api.register(
            RegisterRequest(
                name = name,
                username = username,
                password = password,
                gender = gender.name,
                dateOfBirth = dateOfBirth.toString(),
                weightKg = weightKg,
                heightCm = heightCm,
                dailyStepsGoal = dailyStepsGoal,
                dailyCaloriesGoal = dailyCaloriesGoal
            )
        )
        check(resp.isSuccessful) { "Register failed: HTTP ${resp.code()}" }
        val body = checkNotNull(resp.body()) { "Empty response body" }
        tokenStore.token = body.token
        body.user.toDomain()
    }

    override suspend fun logout() {
        tokenStore.clear()
    }
}
