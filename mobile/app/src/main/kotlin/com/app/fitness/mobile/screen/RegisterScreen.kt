package com.app.fitness.mobile.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.mobile.viewmodel.RegisterViewModel
import com.app.fitness.mobile.components.DumbbellIcon
import com.app.fitness.mobile.components.InputField
import com.app.fitness.Gender

@Composable
fun RegisterScreen(
    viewModel: RegisterViewModel,
    onRegisterSuccess: () -> Unit,
    onGoToLogin: () -> Unit
)
/*@Composable
fun RegisterStartScreen(
    viewModel: RegisterViewModel,
    onNext: () -> Unit,
    onGoToLogin: () -> Unit
)*/
{

    val state by viewModel.state.collectAsState()

    LaunchedEffect(state.isRegistered) {
        if (state.isRegistered) onRegisterSuccess()
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8F8F8))
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {

        Spacer(modifier = Modifier.height(92.dp))

        DumbbellIcon(
            modifier = Modifier.size(202.dp),
            color = Color(0xFF2C2C2C)
        )

        Text(
            text = "Регистрация",
            color = Color(0xFF2C2C2C),
            fontSize = 48.sp,
            letterSpacing = 0.25.sp,
            fontFamily = FontFamily.SansSerif
        )

        Spacer(modifier = Modifier.height(113.dp))

        InputField(
            value = state.name,
            hint = "Name",
            onValueChange = viewModel::onNameChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.username,
            hint = "Email",
            onValueChange = viewModel::onUsernameChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.password,
            hint = "Password",
            onValueChange = viewModel::onPasswordChanged,
            isPassword = true
        )

        Spacer(modifier = Modifier.height(10.dp))

        Row(
            modifier = Modifier.width(328.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Gender.entries.forEach { gender ->
                val isSelected = state.gender == gender
                FilterChip(
                    selected = isSelected,
                    onClick = { viewModel.onGenderChanged(gender) },
                    label = {
                        Text(
                            if (gender == Gender.MALE) "мужской" else "женский"
                        )
                    }
                )
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.dateOfBirth,
            hint = "Date of birth (yyyy-mm-dd)",
            onValueChange = viewModel::onDateOfBirthChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.weightKg,
            hint = "Weight (kg)",
            onValueChange = viewModel::onWeightChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.heightCm,
            hint = "Height (cm)",
            onValueChange = viewModel::onHeightChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.stepsGoal,
            hint = "Steps goal",
            onValueChange = viewModel::onStepsGoalChanged
        )

        Spacer(modifier = Modifier.height(10.dp))

        InputField(
            value = state.caloriesGoal,
            hint = "Calories goal",
            onValueChange = viewModel::onCaloriesGoalChanged
        )

        if (state.error != null) {
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = state.error!!,
                color = MaterialTheme.colorScheme.error
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = viewModel::register,
            enabled = !state.isLoading,
            modifier = Modifier
                .width(328.dp)
                .height(60.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF2C2C2C)
            ),
            shape = RoundedCornerShape(100.dp)
        ) {

            if (state.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp),
                    strokeWidth = 2.dp,
                    color = Color.White
                )
            } else {
                Text(
                    text = "Регистрация",
                    color = Color(0xFFF8F8F8),
                    fontSize = 24.sp,
                    letterSpacing = 0.25.sp,
                    fontFamily = FontFamily.SansSerif
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        TextButton(onClick = onGoToLogin) {
            Text("уже есть аккаунт? войти", color = Color(0xFF2C2C2C))
        }

        Spacer(modifier = Modifier.height(60.dp))
    }
}
