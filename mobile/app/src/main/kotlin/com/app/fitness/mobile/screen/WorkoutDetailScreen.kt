package com.app.fitness.mobile.screen

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

    Scaffold(
        containerColor = BG,
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        state.workout?.title ?: "тренировка",
                        fontFamily = FontFamily.SansSerif,
                        fontSize = 20.sp,
                        color = INK
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) {
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
                "ожидаемый расход: ${state.estimatedCalories.toInt()} ккал",
                fontFamily = FontFamily.SansSerif,
                fontSize = 15.sp,
                color = INK
            )

            Spacer(modifier = Modifier.height(20.dp))

            Text("упражнения", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)

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
                            "${exercise.muscleGroup.toRu()} · ${exercise.durationSeconds}с · отдых ${exercise.restAfterSeconds}с",
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
                    Text("прервать", color = INK, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
                }
            } else {
                Button(
                    onClick = viewModel::startSession,
                    modifier = Modifier.fillMaxWidth().height(56.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = INK),
                    shape = RoundedCornerShape(100.dp)
                ) {
                    Text("начать тренировку", color = BG, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
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
                title = { Text("поздравляем!", fontFamily = FontFamily.SansSerif, color = INK) },
                text = { Text(state.completedMessage!!, fontFamily = FontFamily.SansSerif, color = INK) },
                confirmButton = {
                    TextButton(onClick = onBack) {
                        Text("ок", color = INK, fontFamily = FontFamily.SansSerif)
                    }
                }
            )
        }
    }
}

private fun WorkoutType.toRu() = when (this) {
    WorkoutType.STRENGTH   -> "силовая"
    WorkoutType.CARDIO     -> "кардио"
    WorkoutType.STRETCHING -> "растяжка"
    WorkoutType.YOGA       -> "йога"
    WorkoutType.HIIT       -> "виит"
}

private fun DifficultyLevel.toRu() = when (this) {
    DifficultyLevel.EASY   -> "лёгкий"
    DifficultyLevel.MEDIUM -> "средний"
    DifficultyLevel.HARD   -> "тяжёлый"
}

private fun MuscleGroup.toRu() = when (this) {
    MuscleGroup.CHEST     -> "грудь"
    MuscleGroup.BACK      -> "спина"
    MuscleGroup.ARMS      -> "руки"
    MuscleGroup.ABS       -> "пресс"
    MuscleGroup.GLUTES    -> "ягодицы"
    MuscleGroup.LEGS      -> "ноги"
    MuscleGroup.FULL_BODY -> "всё тело"
}
