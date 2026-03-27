import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';

@Injectable()
export class SessionsService {
  constructor(private db: DatabaseService) {}

  // format a session row into a response object
  private formatSession(row: any) {
    return {
      id: row.id,
      workoutId: row.workout_id,
      startedAt: row.started_at,
      finishedAt: row.finished_at,
      burnedCalories: row.burned_calories,
      completedEarly: row.completed_early,
    };
  }

  async startSession(userId: string, workoutId: string) {
    // make sure the workout exists
    const workoutResult = await this.db.query(
      'SELECT 1 FROM workouts WHERE id = $1',
      [workoutId],
    );
    const workoutNotFound = !workoutResult.rowCount || workoutResult.rowCount === 0;

    if (workoutNotFound) {
      throw new NotFoundException('workout not found');
    }

    const result = await this.db.query(
      `INSERT INTO workout_sessions (user_id, workout_id, started_at)
       VALUES ($1, $2, NOW()) RETURNING *`,
      [userId, workoutId],
    );

    return this.formatSession(result.rows[0]);
  }

  async completeSession(userId: string, sessionId: string, finishedAt?: string) {
    const finished = finishedAt || new Date().toISOString();

    // look up session to get workoutId
    const sessionResult = await this.db.query(
      'SELECT workout_id FROM workout_sessions WHERE id = $1 AND user_id = $2',
      [sessionId, userId],
    );
    if (!sessionResult.rowCount || sessionResult.rowCount === 0) {
      throw new NotFoundException('session not found');
    }
    const workoutId = sessionResult.rows[0].workout_id;

    // look up user weight for calorie calculation
    const userResult = await this.db.query(
      'SELECT weight_kg FROM users WHERE id = $1',
      [userId],
    );
    const weightKg = userResult.rows[0].weight_kg;

    // calculate calories burned based on workout exercises (MET × weight × duration)
    const exercisesResult = await this.db.query(
      `SELECT e.met, e.duration_seconds FROM exercises e
       JOIN workout_exercises we ON we.exercise_id = e.id
       WHERE we.workout_id = $1`,
      [workoutId],
    );
    let total = 0;
    for (const ex of exercisesResult.rows) {
      total += ex.met * weightKg * (ex.duration_seconds / 3600);
    }
    const burnedCalories = Math.round(total * 100) / 100;

    const result = await this.db.query(
      `UPDATE workout_sessions
       SET finished_at = $1, burned_calories = $2, completed_early = FALSE
       WHERE id = $3 AND user_id = $4
       RETURNING *`,
      [finished, burnedCalories, sessionId, userId],
    );

    // also add the burned calories to today's activity
    const date = finished.slice(0, 10);
    await this.db.query(
      `INSERT INTO daily_activities (user_id, date, burned_calories)
       VALUES ($1, $2, $3)
       ON CONFLICT (user_id, date)
       DO UPDATE SET burned_calories = daily_activities.burned_calories + $3`,
      [userId, date, burnedCalories],
    );

    return this.formatSession(result.rows[0]);
  }

  async abandonSession(userId: string, sessionId: string) {
    const result = await this.db.query(
      `UPDATE workout_sessions
       SET finished_at = NOW(), completed_early = TRUE
       WHERE id = $1 AND user_id = $2
       RETURNING *`,
      [sessionId, userId],
    );

    const notFound = !result.rowCount || result.rowCount === 0;
    if (notFound) {
      throw new NotFoundException('session not found');
    }

    return this.formatSession(result.rows[0]);
  }
}
