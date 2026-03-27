package com.app.fitness.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.content.getSystemService
import com.app.fitness.ActivityRepository
import com.app.fitness.StepCounterService
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import java.time.LocalDateTime

/**
 * Android [Service] implementation of [StepCounterService].
 *
 * Runs as a foreground service (required API 26+) so it survives when the
 * screen is off. Registers a listener for [Sensor.TYPE_STEP_COUNTER] which
 * returns a monotonically increasing integer since last reboot.
 *
 * ## Setup in your app
 *
 * 1. Declare in `AndroidManifest.xml`:
 *    ```xml
 *    <service
 *        android:name="com.app.fitness.service.StepCounterServiceImpl"
 *        android:foregroundServiceType="health"
 *        android:exported="false" />
 *    ```
 * 2. Start via [FitnessSdk.startStepCounting]:
 *    ```kotlin
 *    fitnessSdk.startStepCounting(context)
 *    ```
 * 3. Observe steps in your ViewModel:
 *    ```kotlin
 *    fitnessSdk.stepCounterService.stepFlow
 *        .onEach { steps -> _uiState.update { it.copy(steps = steps) } }
 *        .launchIn(viewModelScope)
 *    ```
 */
class StepCounterServiceImpl : Service(), StepCounterService, SensorEventListener {

    // Injected by FitnessSdk via companion property before start
    private lateinit var activityRepository: ActivityRepository

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private val _stepFlow = MutableSharedFlow<Int>(replay = 1)
    override val stepFlow: Flow<Int> = _stepFlow.asSharedFlow()

    private var sensorManager: SensorManager? = null
    private var stepsSentToday: Int = 0

    // ── Service lifecycle ─────────────────────────────────────────────────────

    override fun onCreate() {
        super.onCreate()
        activityRepository = checkNotNull(activityRepositoryRef) {
            "activityRepository not set — call FitnessSdk.startStepCounting() to start the service"
        }
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildForegroundNotification())
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startTracking()
            ACTION_STOP  -> { stopTracking(); stopSelf() }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        stopTracking()
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // ── StepCounterService ────────────────────────────────────────────────────

    override fun startTracking() {
        sensorManager = getSystemService<SensorManager>()
        val sensor = sensorManager?.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        if (sensor != null) {
            sensorManager?.registerListener(this, sensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
    }

    override fun stopTracking() {
        sensorManager?.unregisterListener(this)
        sensorManager = null
    }

    // ── SensorEventListener ───────────────────────────────────────────────────

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_STEP_COUNTER) return
        val totalStepsSinceBoot = event.values[0].toInt()

        serviceScope.launch {
            activityRepository.saveSteps(
                totalStepsSinceBoot = totalStepsSinceBoot,
                timestamp = LocalDateTime.now()
            )
            _stepFlow.emit(totalStepsSinceBoot)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    // ── Helpers ───────────────────────────────────────────────────────────────

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Step Tracking",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Tracks your daily step count in the background"
        }
        getSystemService<NotificationManager>()?.createNotificationChannel(channel)
    }

    private fun buildForegroundNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_compass)
            .setContentTitle("Tracking your steps")
            .setContentText("Step counter is running in the background")
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()

    companion object {
        const val ACTION_START    = "com.app.fitness.ACTION_START_TRACKING"
        const val ACTION_STOP     = "com.app.fitness.ACTION_STOP_TRACKING"
        const val CHANNEL_ID      = "fitness_step_channel"
        const val NOTIFICATION_ID = 2001

        @Volatile internal var activityRepositoryRef: ActivityRepository? = null

        /** Convenience helper — starts the service. */
        fun start(context: Context) {
            context.startForegroundService(
                Intent(context, StepCounterServiceImpl::class.java)
                    .apply { action = ACTION_START }
            )
        }

        /** Convenience helper — stops the service. */
        fun stop(context: Context) {
            context.startService(
                Intent(context, StepCounterServiceImpl::class.java)
                    .apply { action = ACTION_STOP }
            )
        }
    }
}
