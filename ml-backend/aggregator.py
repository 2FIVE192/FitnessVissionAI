from dataclasses import dataclass

SCORE_GAIN_WEIGHT = 15  # score < 15  → weight gain mode
SCORE_MOTIVATE    = 40  # score 15-40 → motivate mode
                        # score >= 40 → real progress mode


@dataclass
class AggregatedStats:
    total_steps: int
    total_calories: float
    period_days: int
    avg_steps_per_day: float
    dominant_activity_type: str
    projected_weight_loss_kg: float
    projected_muscle_gain_kg: float
    projected_weight_gain_kg: float
    bmi_before: float
    bmi_after: float
    prompt_type: str  # 'gain_weight' | 'motivate' | 'real_progress'
    goal: str


def aggregate_activities(
    avg_steps_per_day: int,
    avg_calories_per_day: float,
    activity_type: str,
    period_months: int,
    current_weight_kg: float,
    height_cm: int,
    goal: str,
) -> AggregatedStats:
    period_days    = period_months * 30
    total_steps    = avg_steps_per_day * period_days
    total_calories = avg_calories_per_day * period_days

    # Fat loss from activity (7700 kcal = 1 kg, efficiency factor 0.65)
    fat_loss = round(
        min(total_calories / 7700 * 0.65, current_weight_kg * 0.15), 2
    )

    # Muscle gain (strength training only; cap at 4 kg over any period)
    muscle_gain = round(
        min(0.1 * period_months, 4.0), 2
    ) if activity_type == "strength" else 0.0

    # Weight gain from inactivity
    inactivity_factor = (
        max(0, 1 - avg_steps_per_day / 5000) * 0.6 +
        max(0, 1 - avg_calories_per_day / 200) * 0.4
    )
    projected_weight_gain = round(inactivity_factor * 1.0 * period_months, 2)

    h = height_cm / 100
    bmi_before = round(current_weight_kg / h ** 2, 1)

    # Mode is determined solely by historical activity (steps + calories).
    step_score = min(max((avg_steps_per_day - 3000) / 9000 * 100, 0), 100)
    cal_score  = min(max((avg_calories_per_day - 100) / 500 * 100, 0), 100)
    history_score = step_score * 0.35 + cal_score * 0.30

    if history_score < SCORE_GAIN_WEIGHT:
        prompt_type = "gain_weight"
        bmi_after   = round((current_weight_kg + projected_weight_gain) / h ** 2, 1)
    elif history_score < SCORE_MOTIVATE:
        prompt_type = "motivate"
        bmi_after   = bmi_before
    else:
        prompt_type = "real_progress"
        bmi_after   = round((current_weight_kg - fat_loss + muscle_gain * 1.1) / h ** 2, 1)

    return AggregatedStats(
        total_steps=total_steps,
        total_calories=total_calories,
        period_days=period_days,
        avg_steps_per_day=avg_steps_per_day,
        dominant_activity_type=activity_type,
        projected_weight_loss_kg=fat_loss,
        projected_muscle_gain_kg=muscle_gain,
        projected_weight_gain_kg=projected_weight_gain,
        bmi_before=bmi_before,
        bmi_after=bmi_after,
        prompt_type=prompt_type,
        goal=goal,
    )
