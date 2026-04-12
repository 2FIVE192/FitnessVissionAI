package com.app.fitness.http

import com.app.fitness.http.dto.*
import retrofit2.Response
import retrofit2.http.*

// retrofit service interface — one method per backend endpoint
internal interface FitnessApiService {

    // -- auth --

    @POST("auth/register")
    suspend fun register(@Body body: RegisterRequest): Response<AuthResponse>

    @POST("auth/login")
    suspend fun login(@Body body: LoginRequest): Response<AuthResponse>

    // -- user --

    @GET("user")
    suspend fun getUser(): Response<UserDto>

    @PATCH("user/profile")
    suspend fun updateProfile(@Body body: UpdateProfileRequest): Response<UserDto>

    @PUT("user/goals")
    suspend fun updateGoals(@Body body: UpdateGoalsRequest): Response<UserDto>

    // -- activity --

    @GET("activity/today")
    suspend fun getTodayActivity(): Response<DailyActivityDto>

    @GET("activity/history")
    suspend fun getActivityHistory(@Query("days") days: Int): Response<List<DailyActivityDto>>

    @POST("activity/steps")
    suspend fun saveSteps(@Body body: SaveStepsRequest): Response<DailyActivityDto>

    @POST("activity/manual")
    suspend fun saveManualActivity(@Body body: ManualActivityRequest): Response<DailyActivityDto>

    // -- workouts --

    @GET("workouts")
    suspend fun getAllWorkouts(): Response<List<WorkoutDto>>

    @GET("workouts/recommended")
    suspend fun getRecommendedWorkouts(): Response<List<WorkoutDto>>

    @GET("workouts/favorites")
    suspend fun getFavoriteWorkouts(): Response<List<WorkoutDto>>

    @GET("workouts/{id}")
    suspend fun getWorkoutById(@Path("id") id: String): Response<WorkoutDto>

    @POST("workouts/{id}/favorite")
    suspend fun toggleFavorite(@Path("id") id: String): Response<ToggleFavoriteResponse>

    // -- sessions --

    @POST("sessions/start")
    suspend fun startSession(@Body body: StartSessionRequest): Response<WorkoutSessionDto>

    @PUT("sessions/{id}/complete")
    suspend fun completeSession(
        @Path("id") id: String,
        @Body body: CompleteSessionRequest
    ): Response<WorkoutSessionDto>

    @PUT("sessions/{id}/abandon")
    suspend fun abandonSession(@Path("id") id: String): Response<WorkoutSessionDto>
}
