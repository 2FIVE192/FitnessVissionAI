import { Pool } from 'pg';

// standalone migration script — run with: npm run migrate

const connectionString =
  process.env.DATABASE_URL ||
  'postgresql://fitness:fitness@localhost:5435/fitness_db';

const pool = new Pool({ connectionString });

const migrations = [
  {
    name: '001_initial_schema',
    sql: `
      -- users table holds accounts and their fitness goals
      CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        name TEXT NOT NULL,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        gender TEXT NOT NULL CHECK (gender IN ('MALE', 'FEMALE')),
        date_of_birth DATE NOT NULL,
        weight_kg REAL NOT NULL,
        height_cm REAL NOT NULL,
        daily_steps_goal INT NOT NULL DEFAULT 10000,
        daily_calories_goal INT NOT NULL DEFAULT 500,
        do_not_disturb_until TIMESTAMPTZ,
        do_not_disturb_permanently BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMPTZ DEFAULT NOW()
      );

      -- exercise catalog
      CREATE TABLE IF NOT EXISTS exercises (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        muscle_group TEXT NOT NULL,
        met DOUBLE PRECISION NOT NULL,
        duration_seconds INT NOT NULL,
        rest_after_seconds INT DEFAULT 0,
        image_res_id INT
      );

      -- workout templates made from exercises
      CREATE TABLE IF NOT EXISTS workouts (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        duration_minutes INT NOT NULL,
        is_recommended BOOLEAN DEFAULT FALSE
      );

      -- junction table: which exercises belong to which workout
      CREATE TABLE IF NOT EXISTS workout_exercises (
        workout_id UUID REFERENCES workouts(id) ON DELETE CASCADE,
        exercise_id UUID REFERENCES exercises(id) ON DELETE CASCADE,
        sort_order INT NOT NULL DEFAULT 0,
        PRIMARY KEY (workout_id, exercise_id)
      );

      -- user's bookmarked workouts
      CREATE TABLE IF NOT EXISTS user_favorites (
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        workout_id UUID REFERENCES workouts(id) ON DELETE CASCADE,
        PRIMARY KEY (user_id, workout_id)
      );

      -- one row per user per day for steps/calories/distance
      CREATE TABLE IF NOT EXISTS daily_activities (
        id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        steps INT DEFAULT 0,
        burned_calories DOUBLE PRECISION DEFAULT 0,
        distance_km DOUBLE PRECISION DEFAULT 0,
        UNIQUE (user_id, date)
      );

      -- weight log entries
      CREATE TABLE IF NOT EXISTS weight_entries (
        id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        date DATE NOT NULL,
        weight_kg REAL NOT NULL,
        UNIQUE (user_id, date)
      );

      -- completed or abandoned workout sessions
      CREATE TABLE IF NOT EXISTS workout_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES users(id) ON DELETE CASCADE,
        workout_id UUID REFERENCES workouts(id) ON DELETE CASCADE,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        finished_at TIMESTAMPTZ,
        burned_calories DOUBLE PRECISION DEFAULT 0,
        completed_early BOOLEAN DEFAULT FALSE
      );
    `,
  },
  {
    name: '002_remove_unused_schema',
    sql: `
      -- drop dead columns from users
      ALTER TABLE users
        DROP COLUMN IF EXISTS do_not_disturb_until,
        DROP COLUMN IF EXISTS do_not_disturb_permanently;

      -- drop dead column from exercises
      ALTER TABLE exercises
        DROP COLUMN IF EXISTS image_res_id;

      -- drop unused weight tracking table
      DROP TABLE IF EXISTS weight_entries;
    `,
  },
];

async function migrate() {
  const client = await pool.connect();
  try {
    // migration tracking table
    await client.query(`
      CREATE TABLE IF NOT EXISTS fitness_migrations (
        id SERIAL PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        applied_at TIMESTAMPTZ DEFAULT NOW()
      );
    `);

    for (const m of migrations) {
      const exists = await client.query(
        'SELECT 1 FROM fitness_migrations WHERE name = $1',
        [m.name],
      );

      const alreadyApplied = exists.rowCount && exists.rowCount > 0;
      if (alreadyApplied) {
        console.log(`  ok ${m.name} (already applied)`);
        continue;
      }

      await client.query(m.sql);
      await client.query(
        'INSERT INTO fitness_migrations (name) VALUES ($1)',
        [m.name],
      );
      console.log(`  ok ${m.name} (applied)`);
    }

    console.log('migrations complete.');
  } finally {
    client.release();
    await pool.end();
  }
}

migrate().catch((err) => {
  console.error('migration failed:', err);
  process.exit(1);
});
