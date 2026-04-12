package com.app.fitness.mobile.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AddCircleOutline
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.mobile.viewmodel.HomeViewModel
import java.text.SimpleDateFormat
import java.util.*

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)

@Composable
fun HomeScreen(viewModel: HomeViewModel) {
    val state by viewModel.state.collectAsState()

    // Manual activity input dialog
    if (state.isManualSheetVisible) {
        val fieldColors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = INK, unfocusedBorderColor = INK_MUTED,
            focusedLabelColor = INK, unfocusedLabelColor = INK_MUTED,
            cursorColor = INK, focusedTextColor = INK, unfocusedTextColor = INK
        )
        AlertDialog(
            onDismissRequest = viewModel::hideManualSheet,
            title = { Text("Добавить активность", fontFamily = FontFamily.SansSerif, color = INK) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = state.manualSteps,
                        onValueChange = viewModel::onManualStepsChanged,
                        label = { Text("Шаги (необязательно)") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        singleLine = true,
                        colors = fieldColors
                    )
                    OutlinedTextField(
                        value = state.manualCalories,
                        onValueChange = viewModel::onManualCaloriesChanged,
                        label = { Text("Калории сожжено") },
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                        singleLine = true,
                        colors = fieldColors
                    )
                }
            },
            confirmButton = {
                Button(
                    onClick = viewModel::saveManualActivity,
                    enabled = !state.isSubmittingManual && state.manualCalories.toFloatOrNull() != null,
                    colors = ButtonDefaults.buttonColors(containerColor = INK)
                ) { Text("Сохранить", color = BG) }
            },
            dismissButton = {
                TextButton(onClick = viewModel::hideManualSheet) {
                    Text("Отмена", color = INK)
                }
            }
        )
    }

    if (state.isLoading) {
        Box(
            Modifier.fillMaxSize().background(BG),
            contentAlignment = Alignment.Center
        ) {
            CircularProgressIndicator(color = INK)
        }
        return
    }

    val user = state.user
    val activity = state.activity

    val greeting = if (user != null) "Привет, ${user.name.split(" ").first()}" else "Привет"

    val dateTime = remember {
        SimpleDateFormat("dd MMMM HH:mm", Locale("ru"))
    }.format(Date())

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
            .padding(horizontal = 24.dp)
    ) {

        Spacer(modifier = Modifier.height(16.dp))

        Box(modifier = Modifier.fillMaxWidth()) {

            Column(modifier = Modifier.align(Alignment.TopStart)) {
                Text(
                    text = greeting,
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 36.sp,
                    letterSpacing = 0.25.sp,
                    color = INK
                )
                Text(
                    text = dateTime,
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 15.sp,
                    color = INK_MUTED
                )
            }

            Row(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                IconButton(
                    onClick = viewModel::showManualSheet,
                    modifier = Modifier.size(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.AddCircleOutline,
                        contentDescription = "Добавить активность",
                        tint = INK
                    )
                }
                IconButton(
                    onClick = viewModel::loadData,
                    modifier = Modifier.size(24.dp)
                ) {
                    Icon(
                        imageVector = Icons.Outlined.Refresh,
                        contentDescription = "Обновить",
                        tint = INK
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(45.dp))

        val steps = activity?.steps ?: 0
        val stepsGoal = user?.dailyStepsGoal ?: 10000
        val stepsProgress = if (stepsGoal > 0) steps.toFloat() / stepsGoal else 0f

        Card(
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, INK, RoundedCornerShape(12.dp)),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Column(modifier = Modifier.padding(20.dp)) {
                Text("Шаги", fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK, letterSpacing = 0.25.sp)
                Spacer(modifier = Modifier.height(10.dp))
                Text("$steps / $stepsGoal", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK)
            }
        }

        Spacer(modifier = Modifier.height(15.dp))

        val burnedCals = activity?.burnedCalories ?: 0.0
        val caloriesGoal = user?.dailyCaloriesGoal ?: 500
        val distance = activity?.distanceKm ?: 0.0

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(15.dp)
        ) {
            Card(
                modifier = Modifier
                    .weight(1f)
                    .border(1.dp, INK, RoundedCornerShape(12.dp)),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text("Калории", fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK, letterSpacing = 0.25.sp)
                    Spacer(modifier = Modifier.height(10.dp))
                    Text("${burnedCals.toInt()} / $caloriesGoal", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK)
                    Text("ккал сожжено", fontFamily = FontFamily.SansSerif, fontSize = 12.sp, color = INK_MUTED)
                }
            }

            Card(
                modifier = Modifier
                    .weight(1f)
                    .border(1.dp, INK, RoundedCornerShape(12.dp)),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text("Расстояние", fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK, letterSpacing = 0.25.sp)
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(String.format("%.2f", distance), fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK)
                    Text("км", fontFamily = FontFamily.SansSerif, fontSize = 12.sp, color = INK_MUTED)
                }
            }
        }

        Spacer(modifier = Modifier.height(15.dp))

        if (user != null) {
            val heightMeters = user.heightCm / 100f
            val bmi = user.weightKg / (heightMeters * heightMeters)
            val bmiCategory = when {
                bmi < 18.5 -> "недостаток веса"
                bmi < 25   -> "норма"
                bmi < 30   -> "избыток веса"
                else       -> "ожирение"
            }

            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, INK, RoundedCornerShape(12.dp)),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(modifier = Modifier.padding(20.dp)) {
                    Text("ИМТ", fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK, letterSpacing = 0.25.sp)
                    Spacer(modifier = Modifier.height(10.dp))
                    Text(
                        "${String.format("%.1f", bmi)} — $bmiCategory",
                        fontFamily = FontFamily.SansSerif,
                        fontSize = 15.sp,
                        color = INK
                    )
                }
            }
        }

        if (state.error != null) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = state.error!!,
                color = MaterialTheme.colorScheme.error,
                fontFamily = FontFamily.SansSerif,
                fontSize = 12.sp
            )
        }
    }
}
