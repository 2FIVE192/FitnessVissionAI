package com.app.fitness.http

import com.app.fitness.*
import com.app.fitness.http.DtoMapper.toDomain
import com.app.fitness.http.dto.CompleteSessionRequest
import com.app.fitness.http.dto.StartSessionRequest
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/** HTTP implementation of [WorkoutSessionRepository]. */
class WorkoutSessionRepositoryImpl internal constructor(
    private val client: FitnessApiClient
) : WorkoutSessionRepository {

    private val api get() = client.service

    override suspend fun startSession(workoutId: String): WorkoutSession {
        val resp = api.startSession(StartSessionRequest(workoutId))
        check(resp.isSuccessful) { "startSession failed: HTTP ${resp.code()}" }
        return checkNotNull(resp.body()).toDomain()
    }

    override suspend fun completeSession(
        sessionId: String,
        finishedAt: LocalDateTime
    ): Result<WorkoutSession> = runCatching {
        val resp = api.completeSession(
            sessionId,
            CompleteSessionRequest(
                finishedAt = finishedAt.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + "Z"
            )
        )
        check(resp.isSuccessful) { "completeSession failed: HTTP ${resp.code()}" }
        checkNotNull(resp.body()).toDomain()
    }

    override suspend fun abandonSession(
        sessionId: String,
        finishedAt: LocalDateTime
    ): Result<WorkoutSession> = runCatching {
        val resp = api.abandonSession(sessionId)
        check(resp.isSuccessful) { "abandonSession failed: HTTP ${resp.code()}" }
        checkNotNull(resp.body()).toDomain()
    }
}
