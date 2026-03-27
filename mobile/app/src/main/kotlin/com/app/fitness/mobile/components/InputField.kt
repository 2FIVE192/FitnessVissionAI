package com.app.fitness.mobile.components

import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun InputField(
    value: String,
    hint: String,
    onValueChange: (String) -> Unit,
    isPassword: Boolean = false
) {

    Box(
        modifier = Modifier
            .width(328.dp)
            .height(60.dp)
            .border(
                width = 2.dp,
                color = Color(0xFF2C2C2C),
                shape = RoundedCornerShape(100.dp)
            ),
        contentAlignment = Alignment.CenterStart
    ) {

        BasicTextField(
            value = value,
            onValueChange = onValueChange,
            visualTransformation =
                if (isPassword) PasswordVisualTransformation()
                else androidx.compose.ui.text.input.VisualTransformation.None,
            singleLine = true,
            textStyle = TextStyle(
                color = Color(0xFF2C2C2C),
                fontSize = 15.sp,
                fontFamily = FontFamily.SansSerif
            ),
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    start = 30.dp,
                    end = 30.dp,
                    top = 20.dp,
                    bottom = 20.dp
                )
        )

        if (value.isEmpty()) {
            androidx.compose.material3.Text(
                text = hint,
                color = Color(0x802C2C2C),
                fontSize = 15.sp,
                fontFamily = FontFamily.SansSerif,
                modifier = Modifier.padding(start = 30.dp)
            )
        }
    }
}