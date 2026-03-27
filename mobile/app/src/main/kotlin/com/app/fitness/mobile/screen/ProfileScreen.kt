package com.app.fitness.mobile.screen

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.mobile.viewmodel.ProfileViewModel

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)

@Composable
fun ProfileScreen(
    viewModel: ProfileViewModel,
    onLogout: () -> Unit,
    onEnablePedometer: () -> Unit,
    onDisablePedometer: () -> Unit
) {
    val state by viewModel.state.collectAsState()

    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("fitness_prefs", Context.MODE_PRIVATE) }
    var pedometerEnabled by remember { mutableStateOf(prefs.getBoolean("pedometer_enabled", false)) }

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

    val fieldColors = OutlinedTextFieldDefaults.colors(
        focusedBorderColor = INK,
        unfocusedBorderColor = INK_MUTED,
        focusedLabelColor = INK,
        unfocusedLabelColor = INK_MUTED,
        cursorColor = INK,
        focusedTextColor = INK,
        unfocusedTextColor = INK
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
            .padding(20.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Text(
            text = "профиль",
            fontFamily = FontFamily.SansSerif,
            fontSize = 36.sp,
            letterSpacing = 0.25.sp,
            color = INK
        )

        Spacer(modifier = Modifier.height(16.dp))

        // user info card
        if (user != null) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .border(1.dp, INK, RoundedCornerShape(12.dp)),
                colors = CardDefaults.cardColors(containerColor = Color.White)
            ) {
                Column(modifier = Modifier.padding(16.dp)) {
                    Text(user.name, fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK)
                    Text("@${user.username}", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        "${user.gender.name.lowercase()} · р. ${user.dateOfBirth}",
                        fontFamily = FontFamily.SansSerif,
                        fontSize = 12.sp,
                        color = INK_MUTED
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // body measurements section
        Text("тело", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(modifier = Modifier.height(8.dp))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = state.editWeightKg,
                onValueChange = viewModel::onWeightChanged,
                label = { Text("вес (кг)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                singleLine = true,
                colors = fieldColors,
                modifier = Modifier.weight(1f)
            )
            OutlinedTextField(
                value = state.editHeightCm,
                onValueChange = viewModel::onHeightChanged,
                label = { Text("рост (см)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                singleLine = true,
                colors = fieldColors,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = viewModel::saveProfile,
            enabled = !state.isSaving,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = INK),
            shape = RoundedCornerShape(100.dp)
        ) {
            Text("сохранить профиль", color = BG, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
        }

        Spacer(modifier = Modifier.height(20.dp))

        // goals section
        Text("дневные цели", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(modifier = Modifier.height(8.dp))

        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = state.editStepsGoal,
                onValueChange = viewModel::onStepsGoalChanged,
                label = { Text("шаги") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true,
                colors = fieldColors,
                modifier = Modifier.weight(1f)
            )
            OutlinedTextField(
                value = state.editCaloriesGoal,
                onValueChange = viewModel::onCaloriesGoalChanged,
                label = { Text("калории") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true,
                colors = fieldColors,
                modifier = Modifier.weight(1f)
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = viewModel::saveGoals,
            enabled = !state.isSaving,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            colors = ButtonDefaults.buttonColors(containerColor = INK),
            shape = RoundedCornerShape(100.dp)
        ) {
            Text("сохранить цели", color = BG, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
        }

        // success / error messages
        if (state.successMessage != null) {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = state.successMessage!!,
                color = INK_MUTED,
                fontFamily = FontFamily.SansSerif,
                fontSize = 12.sp
            )
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

        Spacer(modifier = Modifier.height(20.dp))

        // pedometer section
        Text("шагомер", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(modifier = Modifier.height(8.dp))
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .border(1.dp, INK_MUTED, RoundedCornerShape(12.dp)),
            colors = CardDefaults.cardColors(containerColor = Color.White)
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    "Автоматический подсчёт шагов",
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 15.sp,
                    color = INK
                )
                Switch(
                    checked = pedometerEnabled,
                    onCheckedChange = { enabled ->
                        pedometerEnabled = enabled
                        prefs.edit().putBoolean("pedometer_enabled", enabled).apply()
                        if (enabled) onEnablePedometer() else onDisablePedometer()
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(32.dp))

        // logout
        OutlinedButton(
            onClick = {
                viewModel.logout()
                onLogout()
            },
            modifier = Modifier.fillMaxWidth().height(56.dp),
            border = androidx.compose.foundation.BorderStroke(2.dp, INK),
            shape = RoundedCornerShape(100.dp)
        ) {
            Text("выйти", color = INK, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}
