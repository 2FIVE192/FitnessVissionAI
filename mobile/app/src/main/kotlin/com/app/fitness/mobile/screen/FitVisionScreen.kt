package com.app.fitness.mobile.screen

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.GenerationMode
import com.app.fitness.mobile.viewmodel.FitVisionViewModel

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)

private val COLOR_GAIN_WEIGHT   = Color(0xFFFF4444)
private val COLOR_MOTIVATE      = Color(0xFFFFAA00)
private val COLOR_REAL_PROGRESS = Color(0xFF00E676)

@Composable
fun FitVisionScreen(viewModel: FitVisionViewModel) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current

    val photoPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        val bytes = context.contentResolver.openInputStream(uri)?.readBytes() ?: return@rememberLauncherForActivityResult
        viewModel.onPhotoSelected(bytes)
    }

    if (state.isLoading) {
        Box(Modifier.fillMaxSize().background(BG), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(color = INK)
        }
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
            .padding(horizontal = 24.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Spacer(Modifier.height(16.dp))

        Text(
            "FitVision",
            fontFamily = FontFamily.SansSerif,
            fontSize = 36.sp,
            color = INK,
            letterSpacing = 0.25.sp
        )
        Text(
            "Прогноз вашего облика через ${state.periodMonths} мес.",
            fontFamily = FontFamily.SansSerif,
            fontSize = 15.sp,
            color = INK_MUTED
        )

        Spacer(Modifier.height(24.dp))

        // ── Photo section ──────────────────────────────────────────────────

        if (state.selectedPhotoBitmap == null) {
            Button(
                onClick = { photoPickerLauncher.launch("image/*") },
                modifier = Modifier.fillMaxWidth().height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = INK),
                shape = RoundedCornerShape(100.dp)
            ) {
                Text("Выбрать фото из галереи", color = BG, fontFamily = FontFamily.SansSerif)
            }
        } else {
            val selectedPhotoBitmap = state.selectedPhotoBitmap!!
            val modeColor = when (state.generationMode) {
                GenerationMode.GAIN_WEIGHT   -> COLOR_GAIN_WEIGHT
                GenerationMode.REAL_PROGRESS -> COLOR_REAL_PROGRESS
                else                         -> COLOR_MOTIVATE
            }

            // Before
            Image(
                bitmap = selectedPhotoBitmap,
                contentDescription = "До",
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(3f / 4f)
                    .clip(RoundedCornerShape(12.dp)),
                contentScale = ContentScale.Crop
            )
            Spacer(Modifier.height(4.dp))
            Text("До", fontSize = 12.sp, color = INK_MUTED, fontFamily = FontFamily.SansSerif)

            Spacer(Modifier.height(16.dp))

            // After
            val resultBitmap = state.resultBitmap
            if (resultBitmap != null) {
                Image(
                    bitmap = resultBitmap,
                    contentDescription = "После",
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(3f / 4f)
                        .clip(RoundedCornerShape(12.dp))
                        .border(3.dp, modeColor, RoundedCornerShape(12.dp)),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(3f / 4f)
                        .background(Color(0xFFE0E0E0), RoundedCornerShape(12.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text("После", color = INK_MUTED, fontSize = 14.sp, fontFamily = FontFamily.SansSerif)
                }
            }
            Spacer(Modifier.height(4.dp))
            val modeLabel = when (state.generationMode) {
                GenerationMode.GAIN_WEIGHT   -> "Набор веса"
                GenerationMode.MOTIVATE      -> "Потенциал"
                GenerationMode.REAL_PROGRESS -> "Реальный прогресс"
                null                         -> "После"
            }
            Text(
                modeLabel,
                fontSize = 12.sp,
                fontFamily = FontFamily.SansSerif,
                color = if (state.generationMode != null) modeColor else INK_MUTED
            )

            Spacer(Modifier.height(8.dp))
            TextButton(
                onClick = { photoPickerLauncher.launch("image/*") },
                modifier = Modifier.align(Alignment.CenterHorizontally)
            ) {
                Text("Сменить фото", color = INK_MUTED, fontFamily = FontFamily.SansSerif, fontSize = 13.sp)
            }
        }

        Spacer(Modifier.height(24.dp))

        // ── Period ─────────────────────────────────────────────────────────

        Text("Период прогноза", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(1 to "1 мес", 3 to "3 мес", 6 to "6 мес").forEach { (months, label) ->
                FilterChip(
                    selected = state.periodMonths == months,
                    onClick = { viewModel.onPeriodMonthsChanged(months) },
                    label = { Text(label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }

        Spacer(Modifier.height(20.dp))

        // ── Activity type ──────────────────────────────────────────────────

        Text("Тип активности", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("walking" to "Ходьба", "running" to "Бег").forEach { (type, label) ->
                FilterChip(
                    selected = state.activityType == type,
                    onClick = { viewModel.onActivityTypeChanged(type) },
                    label = { Text(label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }
        Spacer(Modifier.height(4.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("strength" to "Силовые", "cycling" to "Велоспорт").forEach { (type, label) ->
                FilterChip(
                    selected = state.activityType == type,
                    onClick = { viewModel.onActivityTypeChanged(type) },
                    label = { Text(label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }

        Spacer(Modifier.height(20.dp))

        // ── Goal ───────────────────────────────────────────────────────────

        Text("Цель", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(
                "lose_weight" to "Похудеть",
                "gain_muscle" to "Мышцы",
                "maintain"    to "Поддержать форму"
            ).forEach { (goal, label) ->
                FilterChip(
                    selected = state.goal == goal,
                    onClick = { viewModel.onGoalChanged(goal) },
                    label = { Text(label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }

        Spacer(Modifier.height(28.dp))

        // ── Generate button ────────────────────────────────────────────────

        if (state.isGenerating) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(modifier = Modifier.size(24.dp), color = INK, strokeWidth = 2.dp)
                Spacer(Modifier.width(12.dp))
                Text(
                    "Генерация... (~30 сек)",
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 15.sp,
                    color = INK_MUTED
                )
            }
        } else {
            Button(
                onClick = viewModel::generate,
                enabled = state.selectedPhotoBytes != null && state.user != null,
                modifier = Modifier.fillMaxWidth().height(56.dp),
                colors = ButtonDefaults.buttonColors(containerColor = INK),
                shape = RoundedCornerShape(100.dp)
            ) {
                Text("Сгенерировать прогноз", color = BG, fontFamily = FontFamily.SansSerif, fontSize = 16.sp)
            }
        }

        if (state.error != null) {
            Spacer(Modifier.height(8.dp))
            Text(
                state.error!!,
                color = MaterialTheme.colorScheme.error,
                fontFamily = FontFamily.SansSerif,
                fontSize = 12.sp
            )
        }

        Spacer(Modifier.height(32.dp))
    }
}
