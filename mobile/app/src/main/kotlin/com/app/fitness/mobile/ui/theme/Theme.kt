package com.app.fitness.mobile.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val LightColorScheme = lightColorScheme(
    primary = Blue500,
    primaryContainer = Blue700,
    secondary = Orange500,
    tertiary = Green500,
    error = Red400,
    background = Grey100,
    onBackground = Grey800
)

@Composable
fun FitnessTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = LightColorScheme,
        content = content
    )
}
