package com.app.fitness.http

import okhttp3.OkHttpClient
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

internal class FitVisionApiClient(mlBaseUrl: String) {

    val service: FitVisionApiService by lazy {
        buildRetrofit(mlBaseUrl).create(FitVisionApiService::class.java)
    }

    private fun buildRetrofit(baseUrl: String): Retrofit {
        val okHttp = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            // generation takes ~10-30 s on GPU; allow plenty of headroom
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(60, TimeUnit.SECONDS)
            .build()

        val url = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"

        return Retrofit.Builder()
            .baseUrl(url)
            .client(okHttp)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
    }
}
