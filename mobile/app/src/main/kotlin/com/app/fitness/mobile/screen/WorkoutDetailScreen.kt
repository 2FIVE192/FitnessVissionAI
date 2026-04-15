package com.app.fitness.mobile.screen

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.DifficultyLevel
import com.app.fitness.MuscleGroup
import com.app.fitness.WorkoutType
import com.app.fitness.mobile.viewmodel.WorkoutDetailViewModel

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WorkoutDetailScreen(
    viewModel: WorkoutDetailViewModel,
    onBack: () -> Unit
) {
    val state by viewModel.state.collectAsState()

    // Show confirmation before abandoning via Back gesture or nav icon
    var showAbandonDialog by remember { mutableStateOf(false) }
    // Navigate back only after abandonSession() has finished (isSessionActive flips to false)
    var navigateBackAfterAbandon by remember { mutableStateOf(false) }
    LaunchedEffect(state.isSessionActive) {
        if (navigateBackAfterAbandon && !state.isSessionActive) onBack()
    }

    BackHandler(enabled = state.isSessionActive) { showAbandonDialog = true }

    if (showAbandonDialog) {
        AlertDialog(
            onDismissRequest = { showAbandonDialog = false },
            containerColor = BG,
            title = { Text("Прервать тренировку?", fontFamily = FontFamily.SansSerif, color = INK) },
            text = { Text("Прогресс будет сохранён как прерванная тренировка.", fontFamily = FontFamily.SansSerif, color = INK) },
            confirmButton = {
                TextButton(onClick = {
                    showAbandonDialog = false
                    navigateBackAfterAbandon = true
                    viewModel.abandonSession()
                }) {
                    Text("Прервать", color = MaterialTheme.colorScheme.error, fontFamily = FontFamily.SansSerif)
                }
            },
            dismissButton = {
                TextButton(onClick = { showAbandonDialog = false }) {
                    Text("Продолжить", color = INK, fontFamily = FontFamily.SansSerif)
                }
            }
        )
    }

    Scaffold(
        containerColor = BG,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        state.workout?.title ?: "Тренировка",
                        fontFamily = FontFamily.SansSerif,
                        fontSize = 20.sp,
                        color = INK
                    )
                },
                navigationIcon = {
                    IconButton(onClick = { if (state.isSessionActive) showAbandonDialog = true else onBack() }) {
                        Icon(
                            Icons.AutoMirrored.Filled.ArrowBack,
                            contentDescription = "назад",
                            tint = INK
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = BG,
                    navigationIconContentColor = INK,
                    titleContentColor = INK
                )
            )
        }
    ) { padding ->
        if (state.isLoading) {
            Box(
                Modifier.fillMaxSize().padding(padding),
                contentAlignment = Alignment.Center
            ) {
                CircularProgressIndicator(color = INK)
            }
            return@Scaffold
        }

        val workout = state.workout ?: return@Scaffold

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(16.dp)
                .verticalScroll(rememberScrollState())
        ) {
            // workout info
            val totalSec = workout.exercises.sumOf { it.durationSeconds + it.restAfterSeconds }
            val durationDisplay = if (totalSec % 60 == 0) "${totalSec / 60} мин"
                                  else "${totalSec / 60} м ${totalSec % 60} с"
            val infoLine = "${workout.type.toRu()} · ${workout.difficulty.toRu()} · $durationDisplay"
            Text(infoLine, fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                "Ожидаемый расход: ${state.estimatedCalories.toInt()} ккал",
                fontFamily = FontFamily.SansSerif,
                fontSize = 15.sp,
                color = INK
            )

            Spacer(modifier = Modifier.height(20.dp))

            Text("Упражнения", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)

            Spacer(modifier = Modifier.height(8.dp))

            workout.exercises.forEachIndexed { index, exercise ->
                val isActive = state.isSessionActive && index == state.currentExerciseIndex
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                        .border(
                            1.dp,
                            INK,
                            RoundedCornerShape(12.dp)
                        ),
                    colors = CardDefaults.cardColors(
                        containerColor = if (isActive) INK else Color.White
                    )
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(
                            "${index + 1}. ${exercise.title}",
                            fontFamily = FontFamily.SansSerif,
                            fontSize = 14.sp,
                            color = if (isActive) BG else INK
                        )
                        Text(
                            "${exercise.muscleGroup.toRu()} · ${exercise.durationSeconds}с · Отдых ${exercise.restAfterSeconds}с",
                            fontFamily = FontFamily.SansSerif,
                            fontSize = 12.sp,
                            color = if (isActive) Color(0xCCF8F8F8) else INK_MUTED
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(24.dp))

            // session controls
            if (state.isSessionActive) {
                val minutes = state.elapsedSeconds / 60
                val seconds = state.elapsedSeconds % 60
                Text(
                    "%02d:%02d".format(minutes, seconds),
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 48.sp,
                    color = INK
                )
                Spacer(modifier = Modifier.height(12.dp))

                OutlinedButton(
                    onClick = viewModel::abandonSession,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    border = androidx.compose.foundation.BorderStroke(2.dp, INK),
                    shape = RoundedCornerShape(100.dp)
                ) {
                    Text("Прервать", color = INK, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
                }
            } else {
                Button(
                    onClick = viewModel::startSession,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = INK),
                    shape = RoundedCornerShape(100.dp)
                ) {
                    Text("Начать тренировку", color = BG, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
                }
            }

            if (state.error != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = state.error!!,
                    color = MaterialTheme.colorScheme.error,
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 12.sp
                )
            }
        }

        if (state.isCompleted && state.completedMessage != null) {
            AlertDialog(
                onDismissRequest = {},
                containerColor = BG,
                title = { Text("Поздравляем!", fontFamily = FontFamily.SansSerif, color = INK) },
                text = { Text(state.completedMessage!!, fontFamily = FontFamily.SansSerif, color = INK) },
                confirmButton = {
                    TextButton(onClick = onBack) {
                        Text("ОК", color = INK, fontFamily = FontFamily.SansSerif)
                    }
                }
            )
        }
    }
}

private fun WorkoutType.toRu() = when (this) {
    WorkoutType.STRENGTH   -> "Силовая"
    WorkoutType.CARDIO     -> "Кардио"
    WorkoutType.STRETCHING -> "Растяжка"
    WorkoutType.YOGA       -> "Йога"
    WorkoutType.HIIT       -> "ВИИТ"
}

private fun DifficultyLevel.toRu() = when (this) {
    DifficultyLevel.EASY   -> "Лёгкий"
    DifficultyLevel.MEDIUM -> "Средний"
    DifficultyLevel.HARD   -> "Тяжёлый"
}

private fun MuscleGroup.toRu() = when (this) {
    MuscleGroup.CHEST     -> "Грудь"
    MuscleGroup.BACK      -> "Спина"
    MuscleGroup.ARMS      -> "Руки"
    MuscleGroup.ABS       -> "Пресс"
    MuscleGroup.GLUTES    -> "Ягодицы"
    MuscleGroup.LEGS      -> "Ноги"
    MuscleGroup.FULL_BODY -> "Всё тело"
}
