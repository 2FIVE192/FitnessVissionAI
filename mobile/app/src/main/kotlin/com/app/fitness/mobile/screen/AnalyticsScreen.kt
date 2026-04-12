package com.app.fitness.mobile.screen

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Refresh
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.DailyActivity
import com.app.fitness.mobile.viewmodel.AnalyticsState
import com.app.fitness.mobile.viewmodel.AnalyticsViewModel
import com.app.fitness.mobile.viewmodel.ChartMetric
import com.app.fitness.mobile.viewmodel.HistoryRange
import kotlin.math.ceil
import kotlin.math.pow

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)
private val BAR_COLOR = Color(0xFF2C2C2C)
private val BAR_EMPTY = Color(0xFFE0E0E0)
private val GRID_LINE = Color(0xFFDDDDDD)

@Composable
fun AnalyticsScreen(viewModel: AnalyticsViewModel) {
    val state by viewModel.state.collectAsState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
            .padding(horizontal = 24.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Spacer(Modifier.height(16.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    "Активность",
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 36.sp,
                    color = INK,
                    letterSpacing = 0.25.sp
                )
                Text(
                    "История ваших показателей",
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 15.sp,
                    color = INK_MUTED
                )
            }
            IconButton(
                onClick = { viewModel.refresh() },
                modifier = Modifier.size(24.dp)
            ) {
                Icon(
                    imageVector = Icons.Outlined.Refresh,
                    contentDescription = "Обновить",
                    tint = INK
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        // ── Metric toggle ──────────────────────────────────────────────────
        Text("Показатель", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            ChartMetric.entries.forEach { metric ->
                val label = when (metric) {
                    ChartMetric.STEPS -> "Шаги"
                    ChartMetric.CALORIES -> "Калории"
                }
                FilterChip(
                    selected = state.metric == metric,
                    onClick = { viewModel.selectMetric(metric) },
                    label = { Text(label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }

        Spacer(Modifier.height(16.dp))

        // ── Range selector ─────────────────────────────────────────────────
        Text("Период", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
        Spacer(Modifier.height(8.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            HistoryRange.entries.forEach { range ->
                FilterChip(
                    selected = state.range == range,
                    onClick = { viewModel.selectRange(range) },
                    label = { Text(range.label, fontFamily = FontFamily.SansSerif) }
                )
            }
        }

        Spacer(Modifier.height(24.dp))

        // ── Chart ──────────────────────────────────────────────────────────
        when {
            state.isLoading -> {
                Box(
                    modifier = Modifier.fillMaxWidth().height(220.dp),
                    contentAlignment = Alignment.Center
                ) {
                    CircularProgressIndicator(color = INK)
                }
            }
            state.error != null -> {
                Box(
                    modifier = Modifier.fillMaxWidth().height(220.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(state.error!!, color = MaterialTheme.colorScheme.error, fontFamily = FontFamily.SansSerif)
                }
            }
            else -> {
                ActivityBarChart(
                    history = state.history,
                    metric = state.metric,
                    modifier = Modifier.fillMaxWidth().height(240.dp)
                )

                Spacer(Modifier.height(20.dp))

                SummaryRow(state)
            }
        }

        Spacer(Modifier.height(32.dp))
    }
}

@Composable
private fun ActivityBarChart(
    history: List<DailyActivity>,
    metric: ChartMetric,
    modifier: Modifier = Modifier
) {
    if (history.isEmpty()) {
        Box(modifier.background(BAR_EMPTY, RoundedCornerShape(12.dp)), contentAlignment = Alignment.Center) {
            Text("Нет данных", color = INK_MUTED, fontFamily = FontFamily.SansSerif)
        }
        return
    }

    val values = history.map { day ->
        when (metric) {
            ChartMetric.STEPS -> day.steps.toFloat()
            ChartMetric.CALORIES -> day.burnedCalories.toFloat()
        }
    }
    val maxValue = values.maxOrNull()?.takeIf { it > 0f } ?: 1f
    val yMax = ceil(maxValue / niceStep(maxValue)) * niceStep(maxValue)
    val ySteps = 4

    val labels = history.map { day ->
        val parts = day.date.toString().split("-")
        if (parts.size == 3) "${parts[2]}.${parts[1]}" else day.date.toString().takeLast(5)
    }

    val labelStride = when {
        history.size <= 7 -> 1
        history.size <= 14 -> 2
        else -> 5
    }

    // All insets defined once in dp — shared by Canvas (converted to px) and Text composables.
    val leftPadDp   = 48.dp
    val bottomPadDp = 28.dp
    val topPadDp    = 8.dp

    val density = LocalDensity.current
    var boxSizePx by remember { mutableStateOf(IntSize.Zero) }

    Box(modifier = modifier.onSizeChanged { boxSizePx = it }) {
        val totalWidthPx  = boxSizePx.width.toFloat()
        val totalHeightPx = boxSizePx.height.toFloat()

        // Skip first frame before size is known
        if (totalWidthPx == 0f || totalHeightPx == 0f) return@Box

        val leftPadPx   = with(density) { leftPadDp.toPx() }
        val bottomPadPx = with(density) { bottomPadDp.toPx() }
        val topPadPx    = with(density) { topPadDp.toPx() }
        val chartW        = totalWidthPx - leftPadPx
        val chartH        = totalHeightPx - bottomPadPx - topPadPx
        val barCount      = values.size
        val totalBarWPx   = chartW / barCount

        // ── Bars + grid lines ──────────────────────────────────────────────
        Canvas(modifier = Modifier.fillMaxSize()) {
            for (i in 0..ySteps) {
                val frac = i.toFloat() / ySteps
                val y = topPadPx + chartH * (1f - frac)
                drawLine(GRID_LINE, Offset(leftPadPx, y), Offset(totalWidthPx, y), strokeWidth = 1f)
            }

            val barPad = totalBarWPx * 0.2f
            val barW   = totalBarWPx - barPad

            values.forEachIndexed { i, v ->
                val x    = leftPadPx + i * totalBarWPx + barPad / 2
                val barH = (v / yMax) * chartH
                val barY = topPadPx + chartH - barH

                if (v == 0f) {
                    drawRoundRect(BAR_EMPTY, Offset(x, topPadPx + chartH - 4f), Size(barW, 4f), CornerRadius(2f))
                } else {
                    drawRoundRect(BAR_COLOR, Offset(x, barY), Size(barW, barH), CornerRadius(4f))
                }
            }
        }

        // ── Y-axis labels — each pinned to its grid line ───────────────────
        // Text height at 9sp ≈ 7 dp — offset upward by half to centre on the line.
        val halfLabelDp = 7.dp / 2
        for (i in 0..ySteps) {
            val frac = i.toFloat() / ySteps
            val yLinePx = topPadPx + chartH * (1f - frac)
            val yLineDp = with(density) { yLinePx.toDp() }
            Text(
                text = formatAxisValue((yMax * i / ySteps).toInt()),
                fontSize = 9.sp,
                color = INK_MUTED,
                fontFamily = FontFamily.SansSerif,
                textAlign = TextAlign.End,
                maxLines = 1,
                modifier = Modifier
                    .width(leftPadDp - 4.dp)
                    .absoluteOffset(y = yLineDp - halfLabelDp)
            )
        }

        // ── X-axis labels — each centred under its bar (or bar group) ──────
        val chartBottomDp = with(density) { (topPadPx + chartH).toDp() }
        labels.forEachIndexed { i, label ->
            if (i % labelStride != 0) return@forEachIndexed
            // Span covers `labelStride` bars; centre of that span is the label anchor.
            val spanPx      = totalBarWPx * labelStride
            val anchorPx    = leftPadPx + i * totalBarWPx + spanPx / 2
            val anchorDp    = with(density) { anchorPx.toDp() }
            val spanDp      = with(density) { spanPx.toDp() }
            Text(
                text = label,
                fontSize = 9.sp,
                color = INK_MUTED,
                fontFamily = FontFamily.SansSerif,
                textAlign = TextAlign.Center,
                maxLines = 1,
                modifier = Modifier
                    .width(spanDp)
                    .absoluteOffset(x = anchorDp - spanDp / 2, y = chartBottomDp + 4.dp)
            )
        }
    }
}

@Composable
private fun SummaryRow(state: AnalyticsState) {
    val history = state.history.filter {
        when (state.metric) {
            ChartMetric.STEPS -> it.steps > 0
            ChartMetric.CALORIES -> it.burnedCalories > 0
        }
    }
    val activeDays = history.size
    val avg = if (activeDays > 0) {
        when (state.metric) {
            ChartMetric.STEPS -> history.sumOf { it.steps.toDouble() } / activeDays
            ChartMetric.CALORIES -> history.sumOf { it.burnedCalories } / activeDays
        }
    } else 0.0

    val total = when (state.metric) {
        ChartMetric.STEPS -> state.history.sumOf { it.steps.toDouble() }
        ChartMetric.CALORIES -> state.history.sumOf { it.burnedCalories }
    }

    val unit = if (state.metric == ChartMetric.STEPS) "шаг." else "ккал"

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        SummaryCard("Среднее/день", "${avg.toInt()} $unit", Modifier.weight(1f))
        SummaryCard("Всего", "${total.toInt()} $unit", Modifier.weight(1f))
        SummaryCard("Активных дней", "$activeDays", Modifier.weight(1f))
    }
}

@Composable
private fun SummaryCard(label: String, value: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        color = Color.White,
        tonalElevation = 1.dp
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(value, fontSize = 16.sp, color = INK, fontFamily = FontFamily.SansSerif)
            Spacer(Modifier.height(2.dp))
            Text(label, fontSize = 11.sp, color = INK_MUTED, fontFamily = FontFamily.SansSerif, textAlign = TextAlign.Center)
        }
    }
}

private fun niceStep(max: Float): Float {
    val raw = max / 4f
    val exp = kotlin.math.floor(kotlin.math.log10(raw.toDouble())).toInt()
    val pow = 10.0.pow(exp.toDouble()).toFloat()
    return when {
        raw / pow <= 1f -> pow
        raw / pow <= 2f -> 2f * pow
        raw / pow <= 5f -> 5f * pow
        else -> 10f * pow
    }
}

private fun formatAxisValue(v: Int): String = when {
    v >= 1_000_000 -> "${v / 1_000_000}M"
    v >= 1_000 -> "${v / 1_000}k"
    else -> "$v"
}
