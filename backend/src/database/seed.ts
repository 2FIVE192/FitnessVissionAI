import { Pool } from 'pg';

// standalone seed script — run with: npm run seed

const connectionString =
  process.env.DATABASE_URL ||
  'postgresql://fitness:fitness@localhost:5435/fitness_db';

const pool = new Pool({ connectionString });

async function seed() {
  const client = await pool.connect();
  try {
    // skip if data already exists
    const existing = await client.query('SELECT COUNT(*) FROM workouts');
    const alreadySeeded = Number(existing.rows[0].count) > 0;

    if (alreadySeeded) {
      console.log('database already seeded, skipping.');
      return;
    }

    console.log('seeding database...');

    // -- exercises --
    const exercises = [
      { title: 'Push-ups', muscle_group: 'CHEST', met: 8.0, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Wide Push-ups', muscle_group: 'CHEST', met: 8.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Diamond Push-ups', muscle_group: 'CHEST', met: 9.0, duration_seconds: 45, rest_after_seconds: 20 },
      { title: 'Chest Dips (bench)', muscle_group: 'CHEST', met: 8.5, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Superman Hold', muscle_group: 'BACK', met: 4.0, duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Reverse Snow Angels', muscle_group: 'BACK', met: 5.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Prone Y Raises', muscle_group: 'BACK', met: 4.5, duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Bicep Curl (bodyweight)', muscle_group: 'ARMS', met: 5.0, duration_seconds: 45, rest_after_seconds: 10 },
      { title: 'Tricep Dips', muscle_group: 'ARMS', met: 7.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Arm Circles', muscle_group: 'ARMS', met: 3.5, duration_seconds: 30, rest_after_seconds: 10 },
      { title: 'Crunches', muscle_group: 'ABS', met: 5.0, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Plank', muscle_group: 'ABS', met: 4.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Mountain Climbers', muscle_group: 'ABS', met: 8.0, duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Leg Raises', muscle_group: 'ABS', met: 5.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Bicycle Crunches', muscle_group: 'ABS', met: 6.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Glute Bridge', muscle_group: 'GLUTES', met: 5.0, duration_seconds: 45, rest_after_seconds: 10 },
      { title: 'Donkey Kicks', muscle_group: 'GLUTES', met: 5.5, duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Fire Hydrants', muscle_group: 'GLUTES', met: 5.0, duration_seconds: 40, rest_after_seconds: 10 },
      { title: 'Squats', muscle_group: 'LEGS', met: 7.0, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Lunges', muscle_group: 'LEGS', met: 7.0, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'Wall Sit', muscle_group: 'LEGS', met: 4.0, duration_seconds: 45, rest_after_seconds: 15 },
      { title: 'Calf Raises', muscle_group: 'LEGS', met: 4.0, duration_seconds: 30, rest_after_seconds: 10 },
      { title: 'Burpees', muscle_group: 'FULL_BODY', met: 10.0, duration_seconds: 30, rest_after_seconds: 20 },
      { title: 'Jumping Jacks', muscle_group: 'FULL_BODY', met: 8.0, duration_seconds: 60, rest_after_seconds: 15 },
      { title: 'High Knees', muscle_group: 'FULL_BODY', met: 9.0, duration_seconds: 30, rest_after_seconds: 15 },
      { title: 'Jump Squats', muscle_group: 'FULL_BODY', met: 9.5, duration_seconds: 30, rest_after_seconds: 20 },
      { title: 'Forward Fold', muscle_group: 'LEGS', met: 2.5, duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Cat-Cow Stretch', muscle_group: 'BACK', met: 2.0, duration_seconds: 30, rest_after_seconds: 5 },
      { title: "Child's Pose", muscle_group: 'BACK', met: 2.0, duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Downward Dog', muscle_group: 'FULL_BODY', met: 3.0, duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Warrior I', muscle_group: 'LEGS', met: 3.0, duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Warrior II', muscle_group: 'LEGS', met: 3.0, duration_seconds: 30, rest_after_seconds: 5 },
      { title: 'Tree Pose', muscle_group: 'LEGS', met: 2.5, duration_seconds: 30, rest_after_seconds: 5 },
    ];

    const exerciseIds: Record<string, string> = {};
    for (const e of exercises) {
      const res = await client.query(
        `INSERT INTO exercises (title, muscle_group, met, duration_seconds, rest_after_seconds)
         VALUES ($1, $2, $3, $4, $5) RETURNING id`,
        [e.title, e.muscle_group, e.met, e.duration_seconds, e.rest_after_seconds],
      );
      exerciseIds[e.title] = res.rows[0].id;
    }
    console.log(`  ${exercises.length} exercises inserted`);

    // -- workouts --
    const workouts = [
      {
        title: 'Chest Blast', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 12, is_recommended: true,
        exercises: ['Push-ups', 'Wide Push-ups', 'Diamond Push-ups', 'Chest Dips (bench)'],
      },
      {
        title: 'Arm Toner', type: 'STRENGTH', difficulty: 'EASY',
        duration_minutes: 8, is_recommended: false,
        exercises: ['Bicep Curl (bodyweight)', 'Tricep Dips', 'Arm Circles'],
      },
      {
        title: 'Leg Day', type: 'STRENGTH', difficulty: 'HARD',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Squats', 'Lunges', 'Wall Sit', 'Calf Raises', 'Jump Squats'],
      },
      {
        title: 'Back Strength', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 10, is_recommended: false,
        exercises: ['Superman Hold', 'Reverse Snow Angels', 'Prone Y Raises'],
      },
      {
        title: 'Glute Builder', type: 'STRENGTH', difficulty: 'EASY',
        duration_minutes: 8, is_recommended: false,
        exercises: ['Glute Bridge', 'Donkey Kicks', 'Fire Hydrants'],
      },
      {
        title: 'Quick Cardio Burn', type: 'CARDIO', difficulty: 'MEDIUM',
        duration_minutes: 10, is_recommended: true,
        exercises: ['Jumping Jacks', 'High Knees', 'Burpees', 'Mountain Climbers'],
      },
      {
        title: 'Fat Burner Express', type: 'CARDIO', difficulty: 'HARD',
        duration_minutes: 12, is_recommended: true,
        exercises: ['Burpees', 'Jump Squats', 'High Knees', 'Mountain Climbers', 'Jumping Jacks'],
      },
      {
        title: 'HIIT Core Crusher', type: 'HIIT', difficulty: 'HARD',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Burpees', 'Mountain Climbers', 'Bicycle Crunches', 'Plank', 'Jump Squats'],
      },
      {
        title: 'Full Body HIIT', type: 'HIIT', difficulty: 'MEDIUM',
        duration_minutes: 15, is_recommended: true,
        exercises: ['Jumping Jacks', 'Squats', 'Push-ups', 'Crunches', 'High Knees'],
      },
      {
        title: 'Morning Stretch', type: 'STRETCHING', difficulty: 'EASY',
        duration_minutes: 7, is_recommended: true,
        exercises: ['Forward Fold', 'Cat-Cow Stretch', "Child's Pose", 'Downward Dog'],
      },
      {
        title: 'Yoga Flow', type: 'YOGA', difficulty: 'EASY',
        duration_minutes: 10, is_recommended: true,
        exercises: ['Downward Dog', 'Warrior I', 'Warrior II', 'Tree Pose', "Child's Pose"],
      },
      {
        title: 'Core Abs Blast', type: 'STRENGTH', difficulty: 'MEDIUM',
        duration_minutes: 12, is_recommended: false,
        exercises: ['Crunches', 'Plank', 'Mountain Climbers', 'Leg Raises', 'Bicycle Crunches'],
      },
    ];

    for (const w of workouts) {
      const res = await client.query(
        `INSERT INTO workouts (title, type, difficulty, duration_minutes, is_recommended)
         VALUES ($1, $2, $3, $4, $5) RETURNING id`,
        [w.title, w.type, w.difficulty, w.duration_minutes, w.is_recommended],
      );
      const workoutId = res.rows[0].id;

      for (let i = 0; i < w.exercises.length; i++) {
        const exId = exerciseIds[w.exercises[i]];
        if (!exId) continue;
        await client.query(
          `INSERT INTO workout_exercises (workout_id, exercise_id, sort_order)
           VALUES ($1, $2, $3)`,
          [workoutId, exId, i],
        );
      }
    }
    console.log(`  ${workouts.length} workouts inserted`);
    console.log('seeding complete.');
  } finally {
    client.release();
    await pool.end();
  }
}

seed().catch((err) => {
  console.error('seeding failed:', err);
  process.exit(1);
});
