package com.app.fitness.mobile.components

import androidx.compose.foundation.Canvas
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color

@Composable
fun DumbbellIcon(
    modifier: Modifier = Modifier,
    color: Color = Color(0xFF2C2C2C)
) {

    Canvas(modifier = modifier) {

        val barWidth = size.width * 0.5f
        val barHeight = size.height * 0.08f

        val plateWidth = size.width * 0.1f
        val plateHeight = size.height * 0.45f

        val centerX = size.width / 2
        val centerY = size.height / 2

        drawRoundRect(
            color = color,
            topLeft = Offset(centerX - barWidth / 2, centerY - barHeight / 2),
            size = Size(barWidth, barHeight),
            cornerRadius = CornerRadius(20f)
        )

        drawRoundRect(
            color = color,
            topLeft = Offset(centerX - barWidth / 2 - plateWidth, centerY - plateHeight / 2),
            size = Size(plateWidth, plateHeight),
            cornerRadius = CornerRadius(20f)
        )

        drawRoundRect(
            color = color,
            topLeft = Offset(centerX - barWidth / 2 - plateWidth * 2, centerY - plateHeight / 2),
            size = Size(plateWidth, plateHeight),
            cornerRadius = CornerRadius(20f)
        )

        drawRoundRect(
            color = color,
            topLeft = Offset(centerX + barWidth / 2, centerY - plateHeight / 2),
            size = Size(plateWidth, plateHeight),
            cornerRadius = CornerRadius(20f)
        )

        drawRoundRect(
            color = color,
            topLeft = Offset(centerX + barWidth / 2 + plateWidth, centerY - plateHeight / 2),
            size = Size(plateWidth, plateHeight),
            cornerRadius = CornerRadius(20f)
        )
    }
}