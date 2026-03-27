package com.app.fitness.http

import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part

internal interface FitVisionApiService {

    @Multipart
    @POST("generate")
    suspend fun generate(
        @Part photo: MultipartBody.Part,
        @Part("gender") gender: RequestBody,
        @Part("age") age: RequestBody,
        @Part("height_cm") heightCm: RequestBody,
        @Part("weight_kg") weightKg: RequestBody,
        @Part("avg_steps_per_day") avgStepsPerDay: RequestBody,
        @Part("avg_calories_per_day") avgCaloriesPerDay: RequestBody,
        @Part("activity_type") activityType: RequestBody,
        @Part("active_days_per_week") activeDaysPerWeek: RequestBody,
        @Part("period_months") periodMonths: RequestBody,
        @Part("goal") goal: RequestBody,
    ): Response<ResponseBody>
}
