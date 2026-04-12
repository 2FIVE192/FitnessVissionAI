package com.app.fitness.http

import com.app.fitness.*
import com.app.fitness.http.DtoMapper.toDomain
import com.app.fitness.http.dto.ManualActivityRequest
import com.app.fitness.http.dto.SaveStepsRequest
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

internal class ActivityRepositoryImpl(
    private val client: FitnessApiClient
) : ActivityRepository {

    private val api get() = client.service

    private var stepBaseline: Int? = null
    private var stepBaselineDate: LocalDate? = null

    override suspend fun getTodayActivity(): DailyActivity {
        val resp = api.getTodayActivity()
        check(resp.isSuccessful) { "getTodayActivity failed: HTTP ${resp.code()}" }
        return checkNotNull(resp.body()).toDomain()
    }

    override suspend fun getActivityHistory(days: Int): List<DailyActivity> {
        val resp = api.getActivityHistory(days)
        check(resp.isSuccessful) { "getActivityHistory failed: HTTP ${resp.code()}" }
        return checkNotNull(resp.body()).map { it.toDomain() }
    }

    override suspend fun saveSteps(totalStepsSinceBoot: Int, timestamp: LocalDateTime) {
        val date = timestamp.toLocalDate()
        if (stepBaselineDate != date) {
            stepBaseline = totalStepsSinceBoot
            stepBaselineDate = date
        }
        val dailySteps = (totalStepsSinceBoot - (stepBaseline ?: totalStepsSinceBoot)).coerceAtLeast(0)
        val ts = timestamp.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME) + "Z"
        api.saveSteps(SaveStepsRequest(totalStepsSinceBoot = dailySteps, timestamp = ts))
    }

    override suspend fun saveManualActivity(steps: Int, caloriesBurned: Float): DailyActivity {
        val resp = api.saveManualActivity(ManualActivityRequest(steps, caloriesBurned))
        check(resp.isSuccessful) { "saveManualActivity failed: HTTP ${resp.code()}" }
        return checkNotNull(resp.body()).toDomain()
    }
}
