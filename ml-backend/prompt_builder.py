from aggregator import AggregatedStats

ACTIVITY_MAP: dict[str, tuple[str, str]] = {
    "running":  ("running, cardio",       "lean athletic build, toned legs"),
    "walking":  ("walking, light cardio", "slim toned body"),
    "strength": ("weight training, gym",  "muscular definition, toned arms, visible abs"),
    "cycling":  ("cycling, cardio",       "lean legs, toned physique"),
}

GOAL_CHANGE: dict[str, dict] = {
    "lose_weight": {
        "real_progress": lambda s: f"lost {s.projected_weight_loss_kg:.1f}kg fat, visibly leaner",
        "motivate":      "fit lean body, potential after regular training",
    },
    "gain_muscle": {
        "real_progress": lambda s: f"gained {s.projected_muscle_gain_kg:.1f}kg muscle, stronger physique",
        "motivate":      "athletic muscular body, potential after gym training",
    },
    "maintain": {
        "real_progress": lambda s: "maintained healthy physique, toned and fit",
        "motivate":      "healthy toned body, potential with consistent activity",
    },
}

NEGATIVE_PROMPT = (
    "deformed body, blurry, bad anatomy, extra limbs, "
    "distorted face, different person, cartoon, lowres, "
    "text, watermark, double face, mutated hands"
)


def build_prompt(
    gender: str,
    age: int,
    height_cm: int,
    stats: AggregatedStats,
    period_months: int,
) -> tuple[str, str, str]:
    """Return (positive_prompt, negative_prompt, label)."""
    g         = "man" if gender == "male" else "woman"
    act, body = ACTIVITY_MAP.get(stats.dominant_activity_type, ("fitness", "toned body"))
    ptype     = stats.prompt_type
    goal      = stats.goal

    if ptype == "gain_weight":
        gain = stats.projected_weight_gain_kg
        bmi_val = stats.bmi_after

        if gain < 1.5:
            weight_desc = "slightly heavier, a bit more body fat"
        elif gain < 3.5:
            weight_desc = f"gained {gain:.1f}kg of body fat, noticeably heavier, softer body"
        else:
            weight_desc = f"gained {gain:.1f}kg of fat, significantly heavier, much more body fat"

        if bmi_val < 25:   physique = "normal weight but gaining fat"
        elif bmi_val < 30: physique = "overweight appearance, excess fat"
        elif bmi_val < 35: physique = "obese appearance, significant excess weight"
        else:              physique = "severely overweight, unhealthy body composition"

        prompt = (
            f"RAW photo, {g}, {age}yo, {height_cm}cm, "
            f"{weight_desc}, {physique}, "
            f"sedentary lifestyle result after {period_months} months, "
            f"same face, same person, "
            f"natural lighting, studio photo, 8k, photorealistic"
        )
        label = "gain_weight"

    elif ptype == "motivate":
        goal_desc = GOAL_CHANGE.get(goal, GOAL_CHANGE["lose_weight"]).get("motivate", "")
        prompt = (
            f"RAW photo, fit {g}, {age}yo, {height_cm}cm, "
            f"{goal_desc}, {body}, "
            f"after consistent {act} training for {period_months} months, "
            f"same face, same person, "
            f"healthy skin, studio photo, 8k, photorealistic"
        )
        label = "motivate"

    else:  # real_progress
        goal_desc_raw = GOAL_CHANGE.get(goal, GOAL_CHANGE["lose_weight"]).get("real_progress")
        goal_desc = goal_desc_raw(stats) if callable(goal_desc_raw) else goal_desc_raw
        prompt = (
            f"RAW photo, fit {g}, {age}yo, {height_cm}cm, "
            f"after {period_months}mo of {act}, "
            f"{goal_desc}, {body}, "
            f"same face, same person, "
            f"healthy skin, studio photo, 8k, photorealistic"
        )
        label = "real_progress"

    return prompt, NEGATIVE_PROMPT, label
