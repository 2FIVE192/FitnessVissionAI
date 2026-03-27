package com.app.fitness.mobile.screen

import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.rotate
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.material3.OutlinedButton
import androidx.compose.ui.text.font.FontWeight
import kotlinx.coroutines.delay
import com.app.fitness.mobile.viewmodel.LoginViewModel
import com.app.fitness.mobile.components.DumbbellIcon


@Composable
fun AuthStartScreen(
    viewModel: LoginViewModel,
    onLoginClick: () -> Unit,
    onRegisterClick: () -> Unit
) {

    var isLoading by remember { mutableStateOf(true) }

    LaunchedEffect(Unit) {
        delay(2000)
        isLoading = false
    }

    val infiniteTransition = rememberInfiniteTransition()

    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            tween(1200, easing = LinearEasing)
        )
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(Color(0xFFF8F8F8))
    ) {

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {

            Spacer(modifier = Modifier.height(210.dp))

            DumbbellIcon(
                modifier = Modifier
                    .size(202.dp)
                    .rotate(if (isLoading) rotation else 0f)
            )

            if (!isLoading) {

                Text(
                    text = "двигайся с нами",
                    fontSize = 24.sp,
                    color = Color(0xFF2C2C2C),
                    fontFamily = FontFamily.SansSerif,
                    fontWeight = FontWeight.Normal
                )

                Spacer(modifier = Modifier.height(193.dp))

                Button(
                    onClick = onLoginClick,
                    modifier = Modifier
                        .width(328.dp)
                        .height(60.dp),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF2C2C2C)
                    ),
                    shape = RoundedCornerShape(100.dp)
                ) {

                    Text(
                        text = "Вход",
                        color = Color(0xFFF8F8F8),
                        fontSize = 24.sp,
                        letterSpacing = 0.25.sp,
                        fontFamily = FontFamily.SansSerif
                    )
                }

                Spacer(modifier = Modifier.height(10.dp))

                OutlinedButton(
                    onClick = onRegisterClick,
                    modifier = Modifier
                        .width(328.dp)
                        .height(60.dp),
                    border = androidx.compose.foundation.BorderStroke(
                        2.dp,
                        Color(0xFF2C2C2C)
                    ),
                    shape = RoundedCornerShape(100.dp)
                ) {

                    Text(
                        text = "Регистрация",
                        color = Color(0xFF2C2C2C),
                        fontSize = 24.sp,
                        letterSpacing = 0.25.sp,
                        fontFamily = FontFamily.SansSerif
                    )
                }
            }
        }
    }
}
