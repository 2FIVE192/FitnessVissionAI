package com.app.fitness.mobile.screen

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.FavoriteBorder
import androidx.compose.material.icons.filled.FilterList
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.app.fitness.*
import com.app.fitness.mobile.viewmodel.WorkoutsViewModel

private val BG = Color(0xFFF8F8F8)
private val INK = Color(0xFF2C2C2C)
private val INK_MUTED = Color(0x802C2C2C)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WorkoutsScreen(
    viewModel: WorkoutsViewModel,
    onWorkoutClick: (String) -> Unit
) {
    val state by viewModel.state.collectAsState()
    var showFilterSheet by remember { mutableStateOf(false) }

    if (state.isLoading) {
        Box(
            Modifier.fillMaxSize().background(BG),
            contentAlignment = Alignment.Center
        ) {
            CircularProgressIndicator(color = INK)
        }
        return
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BG)
    ) {
        // tab row + filter button
        Row(verticalAlignment = Alignment.CenterVertically) {
            val tabTitles = listOf("Все", "Для вас", "Избранное")
            TabRow(
                selectedTabIndex = state.selectedTab,
                modifier = Modifier.weight(1f),
                containerColor = BG,
                contentColor = INK
            ) {
                tabTitles.forEachIndexed { index, title ->
                    Tab(
                        selected = state.selectedTab == index,
                        onClick = { viewModel.selectTab(index) },
                        text = {
                            Text(
                                title,
                                fontFamily = FontFamily.SansSerif,
                                color = if (state.selectedTab == index) INK else INK_MUTED
                            )
                        }
                    )
                }
            }
            IconButton(onClick = { showFilterSheet = true }) {
                Icon(
                    Icons.Default.FilterList,
                    contentDescription = "фильтр",
                    tint = if (!state.filter.isEmpty) INK else INK_MUTED
                )
            }
        }

        val workoutsToShow = state.filteredWorkouts

        if (workoutsToShow.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("тренировки не найдены", fontFamily = FontFamily.SansSerif, fontSize = 15.sp, color = INK_MUTED)
            }
        } else {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                items(workoutsToShow, key = { it.id }) { workout ->
                    WorkoutCard(
                        workout = workout,
                        onClick = { onWorkoutClick(workout.id) },
                        onToggleFavorite = { viewModel.toggleFavorite(workout.id) }
                    )
                }
            }
        }

        if (state.error != null) {
            Text(
                text = state.error!!,
                color = MaterialTheme.colorScheme.error,
                fontFamily = FontFamily.SansSerif,
                fontSize = 12.sp,
                modifier = Modifier.padding(16.dp)
            )
        }
    }

    if (showFilterSheet) {
        FilterBottomSheet(
            current = state.filter,
            onApply = { viewModel.applyFilter(it); showFilterSheet = false },
            onClear = { viewModel.clearFilter(); showFilterSheet = false },
            onDismiss = { showFilterSheet = false }
        )
    }
}


@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun FilterBottomSheet(
    current: WorkoutFilter,
    onApply: (WorkoutFilter) -> Unit,
    onClear: () -> Unit,
    onDismiss: () -> Unit
) {
    var types by remember { mutableStateOf(current.types) }
    var difficulties by remember { mutableStateOf(current.difficulties) }
    var durations by remember { mutableStateOf(current.durations) }
    var muscleGroups by remember { mutableStateOf(current.muscleGroups) }

    val chipColors = FilterChipDefaults.filterChipColors(
        selectedContainerColor = INK,
        selectedLabelColor = BG,
        labelColor = INK
    )
    val chipBorder = FilterChipDefaults.filterChipBorder(
        enabled = true,
        selected = false,
        borderColor = INK
    )
    val chipBorderSelected = FilterChipDefaults.filterChipBorder(
        enabled = true,
        selected = true,
        borderColor = INK,
        selectedBorderColor = INK
    )

    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = BG
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp)
                .padding(bottom = 32.dp)
        ) {
            Text("фильтр тренировок", fontFamily = FontFamily.SansSerif, fontSize = 20.sp, color = INK)
            Spacer(Modifier.height(16.dp))

            FilterSection("тип") {
                WorkoutType.entries.forEach { type ->
                    val sel = type in types
                    FilterChip(
                        selected = sel,
                        onClick = { types = if (sel) types - type else types + type },
                        label = { Text(type.toRu()) },
                        colors = chipColors,
                        border = if (sel) chipBorderSelected else chipBorder
                    )
                }
            }

            FilterSection("сложность") {
                DifficultyLevel.entries.forEach { diff ->
                    val sel = diff in difficulties
                    FilterChip(
                        selected = sel,
                        onClick = { difficulties = if (sel) difficulties - diff else difficulties + diff },
                        label = { Text(diff.toRu()) },
                        colors = chipColors,
                        border = if (sel) chipBorderSelected else chipBorder
                    )
                }
            }

            FilterSection("длительность") {
                DurationRange.entries.forEach { dur ->
                    val label = when (dur) {
                        DurationRange.SHORT    -> "до 3 мин"
                        DurationRange.MEDIUM   -> "3–5 мин"
                        DurationRange.LONG     -> "5–8 мин"
                        DurationRange.EXTENDED -> "8+ мин"
                    }
                    val sel = dur in durations
                    FilterChip(
                        selected = sel,
                        onClick = { durations = if (sel) durations - dur else durations + dur },
                        label = { Text(label) },
                        colors = chipColors,
                        border = if (sel) chipBorderSelected else chipBorder
                    )
                }
            }

            FilterSection("группа мышц") {
                MuscleGroup.entries.forEach { mg ->
                    val sel = mg in muscleGroups
                    FilterChip(
                        selected = sel,
                        onClick = { muscleGroups = if (sel) muscleGroups - mg else muscleGroups + mg },
                        label = { Text(mg.toRu()) },
                        colors = chipColors,
                        border = if (sel) chipBorderSelected else chipBorder
                    )
                }
            }

            Spacer(Modifier.height(16.dp))

            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(
                    onClick = onClear,
                    modifier = Modifier.weight(1f).height(56.dp),
                    border = androidx.compose.foundation.BorderStroke(2.dp, INK),
                    shape = RoundedCornerShape(100.dp)
                ) {
                    Text("сбросить", color = INK, fontFamily = FontFamily.SansSerif)
                }
                Button(
                    onClick = { onApply(WorkoutFilter(types, muscleGroups, difficulties, durations)) },
                    modifier = Modifier.weight(1f).height(56.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = INK),
                    shape = RoundedCornerShape(100.dp)
                ) {
                    Text("применить", color = BG, fontFamily = FontFamily.SansSerif)
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun FilterSection(title: String, content: @Composable FlowRowScope.() -> Unit) {
    Text(title, fontFamily = FontFamily.SansSerif, fontSize = 13.sp, color = INK_MUTED)
    Spacer(Modifier.height(6.dp))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), content = content)
    Spacer(Modifier.height(16.dp))
}

@Composable
private fun WorkoutCard(
    workout: Workout,
    onClick: () -> Unit,
    onToggleFavorite: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, INK, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = workout.title,
                    fontFamily = FontFamily.SansSerif,
                    fontSize = 16.sp,
                    color = INK
                )

                val totalSec = workout.exercises.sumOf { it.durationSeconds + it.restAfterSeconds }
                val durationDisplay = if (totalSec % 60 == 0) "${totalSec / 60} мин"
                                      else "${totalSec / 60} м ${totalSec % 60} с"
                val infoLine = "${workout.type.toRu()} · ${workout.difficulty.toRu()} · $durationDisplay"
                Text(infoLine, fontFamily = FontFamily.SansSerif, fontSize = 12.sp, color = INK_MUTED)
                Text("${workout.exercises.size} упр.", fontFamily = FontFamily.SansSerif, fontSize = 12.sp, color = INK_MUTED)
            }

            IconButton(onClick = onToggleFavorite) {
                val icon = if (workout.isFavorite) Icons.Default.Favorite else Icons.Default.FavoriteBorder
                Icon(icon, contentDescription = "избранное", tint = INK)
            }
        }
    }
}

private fun WorkoutType.toRu() = when (this) {
    WorkoutType.STRENGTH   -> "силовая"
    WorkoutType.CARDIO     -> "кардио"
    WorkoutType.STRETCHING -> "растяжка"
    WorkoutType.YOGA       -> "йога"
    WorkoutType.HIIT       -> "виит"
}

private fun DifficultyLevel.toRu() = when (this) {
    DifficultyLevel.EASY   -> "лёгкий"
    DifficultyLevel.MEDIUM -> "средний"
    DifficultyLevel.HARD   -> "тяжёлый"
}

private fun MuscleGroup.toRu() = when (this) {
    MuscleGroup.CHEST     -> "грудь"
    MuscleGroup.BACK      -> "спина"
    MuscleGroup.ARMS      -> "руки"
    MuscleGroup.ABS       -> "пресс"
    MuscleGroup.GLUTES    -> "ягодицы"
    MuscleGroup.LEGS      -> "ноги"
    MuscleGroup.FULL_BODY -> "всё тело"
}
