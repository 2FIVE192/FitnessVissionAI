package com.app.fitness.http

import android.content.Context
import com.google.gson.*
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.lang.reflect.Type
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit

/**
 * Builds and holds the Retrofit + OkHttp singleton for the Fitness SDK.
 *
 * The [tokenStore] is injected so the auth interceptor can always read the
 * latest token, including after login/register without re-creating the client.
 */
internal class FitnessApiClient(
    context: Context,
    baseUrl: String,
    internal val tokenStore: TokenStore = TokenStore(context),
    enableLogging: Boolean = false
) {

    val service: FitnessApiService by lazy { buildRetrofit(baseUrl, enableLogging).create(FitnessApiService::class.java) }

    private fun buildRetrofit(baseUrl: String, enableLogging: Boolean): Retrofit {
        val okHttp = OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .addInterceptor { chain ->
                val request = chain.request().newBuilder().apply {
                    tokenStore.token?.let { addHeader("Authorization", "Bearer $it") }
                }.build()
                chain.proceed(request)
            }
            .apply {
                if (enableLogging) {
                    addNetworkInterceptor(
                        HttpLoggingInterceptor().apply {
                            level = HttpLoggingInterceptor.Level.BODY
                        }
                    )
                }
            }
            .build()

        val gson = GsonBuilder()
            .registerTypeAdapter(LocalDate::class.java, LocalDateAdapter())
            .registerTypeAdapter(LocalDateTime::class.java, LocalDateTimeAdapter())
            .serializeNulls()
            .create()

        // Normalise trailing slash
        val url = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        // Retrofit requires the base URL to end with /api/ if routes are /api/...
        val apiUrl = if (url.endsWith("api/")) url else "${url}api/"

        return Retrofit.Builder()
            .baseUrl(apiUrl)
            .client(okHttp)
            .addConverterFactory(GsonConverterFactory.create(gson))
            .build()
    }

    // ── Gson type adapters ────────────────────────────────────────────────────

    private class LocalDateAdapter : JsonSerializer<LocalDate>, JsonDeserializer<LocalDate> {
        override fun serialize(src: LocalDate, typeOfSrc: Type, ctx: JsonSerializationContext): JsonElement =
            JsonPrimitive(src.format(DateTimeFormatter.ISO_LOCAL_DATE))

        override fun deserialize(json: JsonElement, typeOfT: Type, ctx: JsonDeserializationContext): LocalDate =
            LocalDate.parse(json.asString.take(10), DateTimeFormatter.ISO_LOCAL_DATE)
    }

    private class LocalDateTimeAdapter : JsonSerializer<LocalDateTime>, JsonDeserializer<LocalDateTime> {
        override fun serialize(src: LocalDateTime, typeOfSrc: Type, ctx: JsonSerializationContext): JsonElement =
            JsonPrimitive(src.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME))

        override fun deserialize(json: JsonElement, typeOfT: Type, ctx: JsonDeserializationContext): LocalDateTime =
            LocalDateTime.parse(
                json.asString.replace("Z", "").take(19),
                DateTimeFormatter.ISO_LOCAL_DATE_TIME
            )
    }
}
