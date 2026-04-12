import { Pool } from 'pg';
import * as bcrypt from 'bcrypt';

// standalone demo-user seed — run with: npm run seed:demo
// Creates one user with a month of consistently strong activity data.
// Safe to re-run: exits early if the username already exists.

const connectionString =
  process.env.DATABASE_URL ||
  'postgresql://fitness:fitness@localhost:5435/fitness_db';

const pool = new Pool({ connectionString });

// Deterministic-ish pseudo-random: returns a float in [min, max]
// using a simple linear congruential generator seeded by day index.
function lcgFloat(seed: number, min: number, max: number): number {
  const a = 1664525;
  const c = 1013904223;
  const m = 2 ** 32;
  const x = ((a * seed + c) % m + m) % m;
  return min + (x / m) * (max - min);
}

async function seedDemoUser() {
  const client = await pool.connect();
  try {
    // Idempotency check
    const existing = await client.query(
      'SELECT id FROM users WHERE username = $1',
      ['aleksey'],
    );
    if (existing.rowCount && existing.rowCount > 0) {
      console.log('demo user "aleksey" already exists — skipping.');
      return;
    }

    console.log('creating demo user...');

    const passwordHash = await bcrypt.hash('demo1234', 10);

    const userRes = await client.query(
      `INSERT INTO users
         (name, username, password_hash, gender, date_of_birth,
          weight_kg, height_cm, daily_steps_goal, daily_calories_goal)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
       RETURNING id`,
      [
        'Алексей Горский',
        'aleksey',
        passwordHash,
        'MALE',
        '1996-03-15',
        82.0,
        181.0,
        10000,
        500,
      ],
    );
    const userId = userRes.rows[0].id;
    console.log(`  user created: id=${userId}`);

    // Build 30 days of activity.
    // The user consistently exceeds both goals every day with natural variation.
    //
    // Steps pattern:
    //   weekdays (Mon-Fri): 11 500 – 15 000  (active commuter + gym)
    //   weekends (Sat-Sun):  10 200 – 12 500  (casual walks, lighter day)
    //   two "hero" days scattered in: ~16 000 steps (long hike)
    //
    // Calories always above goal (500 kcal) and correlated with steps.

    const HEIGHT_CM = 181.0;
    const WEIGHT_KG = 82.0;
    const strideMeters = (HEIGHT_CM * 0.415) / 100; // ~0.751 m

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const heroDays = new Set([4, 18]); // day indices (0 = oldest) that are "hero" days

    const rows: { date: string; steps: number; calories: number; distanceKm: number }[] = [];

    for (let i = 29; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      const dayIndex = 29 - i; // 0 = oldest, 29 = today
      const dayOfWeek = d.getDay(); // 0=Sun, 6=Sat
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
      const isHero = heroDays.has(dayIndex);

      let steps: number;
      if (isHero) {
        steps = Math.round(lcgFloat(dayIndex * 7, 15500, 17000));
      } else if (isWeekend) {
        steps = Math.round(lcgFloat(dayIndex * 7, 10200, 12500));
      } else {
        steps = Math.round(lcgFloat(dayIndex * 7, 11500, 15000));
      }

      // Calories: base from steps + extra "workout" bonus on weekdays
      const stepsCalories = steps * 0.04 * (WEIGHT_KG / 70);
      const workoutBonus = isWeekend ? lcgFloat(dayIndex * 13, 50, 120) : lcgFloat(dayIndex * 13, 80, 220);
      const calories = Math.round((stepsCalories + workoutBonus) * 10) / 10;

      const distanceKm = Math.round((steps * strideMeters / 1000) * 100) / 100;

      const dateStr = d.toISOString().slice(0, 10);
      rows.push({ date: dateStr, steps, calories, distanceKm });
    }

    for (const row of rows) {
      await client.query(
        `INSERT INTO daily_activities (user_id, date, steps, burned_calories, distance_km)
         VALUES ($1, $2, $3, $4, $5)
         ON CONFLICT (user_id, date) DO NOTHING`,
        [userId, row.date, row.steps, row.calories, row.distanceKm],
      );
    }

    const avgSteps = Math.round(rows.reduce((s, r) => s + r.steps, 0) / rows.length);
    const avgCals = Math.round(rows.reduce((s, r) => s + r.calories, 0) / rows.length);

    console.log(`  ${rows.length} days of activity inserted`);
    console.log(`  avg steps/day: ${avgSteps}  |  avg calories/day: ${avgCals}`);
    console.log('');
    console.log('  login: aleksey / demo1234');
    console.log('demo user seed complete.');
  } finally {
    client.release();
    await pool.end();
  }
}

seedDemoUser().catch((err) => {
  console.error('demo seed failed:', err);
  process.exit(1);
});
