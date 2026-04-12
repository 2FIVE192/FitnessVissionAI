import { Injectable } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';

@Injectable()
export class ActivityService {
  constructor(private db: DatabaseService) {}

  async getTodayActivity(userId: string) {
    const today = new Date().toISOString().slice(0, 10);

    const result = await this.db.query(
      'SELECT * FROM daily_activities WHERE user_id = $1 AND date = $2',
      [userId, today],
    );

    const hasNoData = !result.rowCount || result.rowCount === 0;
    if (hasNoData) {
      return { date: today, steps: 0, burnedCalories: 0, distanceKm: 0 };
    }

    const row = result.rows[0];
    return {
      date: row.date,
      steps: row.steps,
      burnedCalories: row.burned_calories,
      distanceKm: row.distance_km,
    };
  }

  async saveSteps(userId: string, totalStepsSinceBoot: number, timestamp: string) {
    const date = timestamp.slice(0, 10);
    const steps = totalStepsSinceBoot;

    // grab user's physical stats for distance/calorie calculation
    const userResult = await this.db.query(
      'SELECT weight_kg, height_cm FROM users WHERE id = $1',
      [userId],
    );
    const user = userResult.rows[0];

    // stride length is roughly 41.5% of height
    const strideMeters = (user.height_cm * 0.415) / 100;
    const distanceKm = (steps * strideMeters) / 1000;

    // rough calorie estimate scaled by body weight
    const calories = steps * 0.04 * (user.weight_kg / 70);

    await this.db.query(
      `INSERT INTO daily_activities (user_id, date, steps, burned_calories, distance_km)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (user_id, date)
       DO UPDATE SET steps = $3, burned_calories = $4, distance_km = $5`,
      [userId, date, steps, calories, distanceKm],
    );

    return { date, steps, burnedCalories: calories, distanceKm };
  }

  async getActivityHistory(userId: string, days: number) {
    const rows = await this.db.query(
      `SELECT date, steps, burned_calories, distance_km
       FROM daily_activities
       WHERE user_id = $1
         AND date >= CURRENT_DATE - ($2 - 1) * INTERVAL '1 day'
       ORDER BY date ASC`,
      [userId, days],
    );

    // Build a full list filling missing days with zeros
    const result: { date: string; steps: number; burnedCalories: number; distanceKm: number }[] = [];
    const today = new Date();
    const rowMap = new Map<string, (typeof rows.rows)[0]>();
    for (const row of rows.rows) {
      rowMap.set(row.date instanceof Date ? row.date.toISOString().slice(0, 10) : String(row.date).slice(0, 10), row);
    }
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      const key = d.toISOString().slice(0, 10);
      const row = rowMap.get(key);
      result.push({
        date: key,
        steps: row ? row.steps : 0,
        burnedCalories: row ? row.burned_calories : 0,
        distanceKm: row ? row.distance_km : 0,
      });
    }
    return result;
  }

  async saveManualActivity(userId: string, steps: number, caloriesBurned: number) {
    const today = new Date().toISOString().slice(0, 10);

    const userResult = await this.db.query(
      'SELECT height_cm FROM users WHERE id = $1',
      [userId],
    );
    const user = userResult.rows[0];
    const strideMeters = (user.height_cm * 0.415) / 100;
    const distanceKm = (steps * strideMeters) / 1000;

    await this.db.query(
      `INSERT INTO daily_activities (user_id, date, steps, burned_calories, distance_km)
       VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (user_id, date)
       DO UPDATE SET
         steps          = daily_activities.steps          + $3,
         burned_calories = daily_activities.burned_calories + $4,
         distance_km    = daily_activities.distance_km    + $5`,
      [userId, today, steps, caloriesBurned, distanceKm],
    );

    return this.getTodayActivity(userId);
  }
}
