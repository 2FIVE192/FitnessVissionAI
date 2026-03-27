from dataclasses import dataclass

SCORE_GAIN_WEIGHT = 15  # score < 15  → weight gain mode
SCORE_MOTIVATE    = 40  # score 15-40 → motivate mode
                        # score >= 40 → real progress mode


@dataclass
class AggregatedStats:
    total_steps: int
    total_calories: float
    active_days: int
    period_days: int
    avg_steps_per_day: float
    dominant_activity_type: str
    projected_weight_loss_kg: float
    projected_muscle_gain_kg: float
    projected_weight_gain_kg: float
    bmi_before: float
    bmi_after: float
    activity_score: float
    score_zone: str
    prompt_type: str  # 'gain_weight' | 'motivate' | 'real_progress'
    goal: str


def aggregate_activities(
    avg_steps_per_day: int,
    avg_calories_per_day: float,
    activity_type: str,
    active_days_per_week: int,
    period_months: int,
    current_weight_kg: float,
    height_cm: int,
    goal: str,
) -> AggregatedStats:
    period_days    = period_months * 30
    active_days    = int(active_days_per_week / 7 * period_days)
    total_steps    = avg_steps_per_day * period_days
    total_calories = avg_calories_per_day * active_days

    # Fat loss from activity (7700 kcal = 1 kg, efficiency factor 0.65)
    fat_loss = round(
        min(total_calories / 7700 * 0.65, current_weight_kg * 0.15), 2
    )

    # Muscle gain (strength training only)
    muscle_gain = round(
        min(active_days_per_week / 7 * 0.7 * period_months, 4.0), 2
    ) if activity_type == "strength" else 0.0

    # Weight gain from inactivity
    inactivity_factor = (
        max(0, 1 - avg_steps_per_day / 5000) * 0.6 +
        max(0, 1 - avg_calories_per_day / 200) * 0.4
    )
    projected_weight_gain = round(inactivity_factor * 1.0 * period_months, 2)

    h = height_cm / 100
    bmi_before = round(current_weight_kg / h ** 2, 1)

    # Activity Score (0–100)
    step_score = min(max((avg_steps_per_day - 3000) / 9000 * 100, 0), 100)
    cal_score  = min(max((avg_calories_per_day - 100) / 500 * 100, 0), 100)
    freq_score = min(active_days_per_week / 7 * 100, 100)
    act_bonus  = {"running": 15, "strength": 12, "cycling": 10, "walking": 0}.get(activity_type, 0)

    activity_score = round(min(
        step_score * 0.35 + cal_score * 0.30 + freq_score * 0.25 + act_bonus,
        100,
    ), 1)

    if activity_score < 20:   score_zone = "inactive"
    elif activity_score < 45: score_zone = "moderate"
    elif activity_score < 70: score_zone = "active"
    else:                     score_zone = "athlete"

    if activity_score < SCORE_GAIN_WEIGHT:
        prompt_type = "gain_weight"
        bmi_after   = round((current_weight_kg + projected_weight_gain) / h ** 2, 1)
    elif activity_score < SCORE_MOTIVATE:
        prompt_type = "motivate"
        bmi_after   = bmi_before
    else:
        prompt_type = "real_progress"
        bmi_after   = round((current_weight_kg - fat_loss + muscle_gain * 1.1) / h ** 2, 1)

    return AggregatedStats(
        total_steps=total_steps,
        total_calories=total_calories,
        active_days=active_days,
        period_days=period_days,
        avg_steps_per_day=avg_steps_per_day,
        dominant_activity_type=activity_type,
        projected_weight_loss_kg=fat_loss,
        projected_muscle_gain_kg=muscle_gain,
        projected_weight_gain_kg=projected_weight_gain,
        bmi_before=bmi_before,
        bmi_after=bmi_after,
        activity_score=activity_score,
        score_zone=score_zone,
        prompt_type=prompt_type,
        goal=goal,
    )
