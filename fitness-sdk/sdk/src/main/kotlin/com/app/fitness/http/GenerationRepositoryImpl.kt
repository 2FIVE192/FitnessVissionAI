package com.app.fitness.http

import com.app.fitness.GenerationMode
import com.app.fitness.GenerationRepository
import com.app.fitness.GenerationRequest
import com.app.fitness.GenerationResult
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.toRequestBody

internal class GenerationRepositoryImpl(
    private val client: FitVisionApiClient
) : GenerationRepository {

    override suspend fun generate(request: GenerationRequest): Result<GenerationResult> {
        return try {
            val photoPart = MultipartBody.Part.createFormData(
                "photo", "photo.jpg",
                request.photoBytes.toRequestBody("image/jpeg".toMediaType())
            )

            fun String.asText() = toRequestBody("text/plain".toMediaType())

            val response = client.service.generate(
                photo               = photoPart,
                gender              = request.gender.asText(),
                age                 = request.age.toString().asText(),
                heightCm            = request.heightCm.toString().asText(),
                weightKg            = request.weightKg.toString().asText(),
                avgStepsPerDay      = request.avgStepsPerDay.toString().asText(),
                avgCaloriesPerDay   = request.avgCaloriesPerDay.toString().asText(),
                activityType        = request.activityType.asText(),
                periodMonths        = request.periodMonths.toString().asText(),
                goal                = request.goal.asText(),
            )

            if (!response.isSuccessful) {
                return Result.failure(Exception("Generation failed: HTTP ${response.code()}"))
            }

            val imageBytes = response.body()?.bytes()
                ?: return Result.failure(Exception("Empty response body"))

            val mode = when (response.headers()["X-Mode"]) {
                "gain_weight"   -> GenerationMode.GAIN_WEIGHT
                "real_progress" -> GenerationMode.REAL_PROGRESS
                else            -> GenerationMode.MOTIVATE
            }

            Result.success(GenerationResult(imageBytes = imageBytes, mode = mode))
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}
