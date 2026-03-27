package com.app.fitness.mobile.screen

import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.mobile.viewmodel.LoginViewModel
import com.app.fitness.mobile.components.DumbbellIcon
import com.app.fitness.mobile.components.InputField

@Composable
fun LoginScreen(
    viewModel: LoginViewModel,
    onLoginSuccess: () -> Unit,
    onGoToRegister: () -> Unit
) {

    val state by viewModel.state.collectAsState()

    // логика перехода (НЕ ТРОГАЕМ)
    LaunchedEffect(state.isLoggedIn) {
        if (state.isLoggedIn) onLoginSuccess()
    }

    val iconOffsetY by animateDpAsState(targetValue = 92.dp)

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8F8F8))
    ) {

        Column(
            modifier = Modifier.fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(iconOffsetY))

            DumbbellIcon(
                modifier = Modifier.size(202.dp)
            )

            Text(
                text = "Вход",
                color = Color(0xFF2C2C2C),
                fontSize = 48.sp,
                letterSpacing = 0.25.sp,
                fontFamily = FontFamily.SansSerif
            )

            Spacer(modifier = Modifier.height(191.dp))


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

            if (state.error != null) {
                Text(
                    text = state.error!!,
                    color = MaterialTheme.colorScheme.error,
                    fontSize = 14.sp
                )

                Spacer(modifier = Modifier.height(10.dp))
            }

            Button(
                onClick = viewModel::login,
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
                        color = Color(0xFFF8F8F8)
                    )
                } else {

                    Text(
                        text = "Войти",
                        color = Color(0xFFF8F8F8),
                        fontSize = 24.sp,
                        letterSpacing = 0.25.sp,
                        fontFamily = FontFamily.SansSerif
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            TextButton(onClick = onGoToRegister) {
                Text(
                    text = "нет аккаунта? зарегистрироваться",
                    color = Color(0xFF2C2C2C)
                )
            }

            Spacer(modifier = Modifier.height(106.dp))
        }
    }
}
