import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';


@Injectable()
export class WorkoutsService {
  constructor(private db: DatabaseService) {}

  // load exercises for a single workout
  private async loadExercises(workoutId: string) {
    const result = await this.db.query(
      `SELECT e.* FROM exercises e
       JOIN workout_exercises we ON we.exercise_id = e.id
       WHERE we.workout_id = $1
       ORDER BY we.sort_order`,
      [workoutId],
    );

    return result.rows.map((e) => ({
      id: e.id,
      title: e.title,
      muscleGroup: e.muscle_group,
      met: e.met,
      durationSeconds: e.duration_seconds,
      restAfterSeconds: e.rest_after_seconds,
    }));
  }

  // format a workout row + exercises + optional favorite flag
  private formatWorkout(row: any, exercises: any[], isFavorite = false) {
    return {
      id: row.id,
      title: row.title,
      type: row.type,
      difficulty: row.difficulty,
      durationMinutes: row.duration_minutes,
      exercises,
      isFavorite,
    };
  }

  // take a list of raw workout rows and hydrate them with exercises + favorites
  private async hydrateWorkouts(rows: any[], userId?: string) {
    // collect user's favorite workout ids
    const favoriteIds = new Set<string>();
    if (userId) {
      const favResult = await this.db.query(
        'SELECT workout_id FROM user_favorites WHERE user_id = $1',
        [userId],
      );
      favResult.rows.forEach((r) => favoriteIds.add(r.workout_id));
    }

    const workouts = [];
    for (const row of rows) {
      const exercises = await this.loadExercises(row.id);
      const isFav = favoriteIds.has(row.id);
      workouts.push(this.formatWorkout(row, exercises, isFav));
    }
    return workouts;
  }

  async getAll(userId?: string) {
    const result = await this.db.query('SELECT * FROM workouts ORDER BY title');
    return this.hydrateWorkouts(result.rows, userId);
  }

  async getRecommended() {
    const result = await this.db.query(
      'SELECT * FROM workouts WHERE is_recommended = TRUE ORDER BY title',
    );
    return this.hydrateWorkouts(result.rows);
  }

  async getFavorites(userId: string) {
    const result = await this.db.query(
      `SELECT w.* FROM workouts w
       JOIN user_favorites uf ON uf.workout_id = w.id
       WHERE uf.user_id = $1
       ORDER BY w.title`,
      [userId],
    );
    return this.hydrateWorkouts(result.rows, userId);
  }

  async getById(workoutId: string) {
    const result = await this.db.query(
      'SELECT * FROM workouts WHERE id = $1',
      [workoutId],
    );

    const notFound = !result.rowCount || result.rowCount === 0;
    if (notFound) {
      throw new NotFoundException('workout not found');
    }

    const exercises = await this.loadExercises(workoutId);
    return this.formatWorkout(result.rows[0], exercises);
  }

  async toggleFavorite(userId: string, workoutId: string) {
    // check if already favorited
    const exists = await this.db.query(
      'SELECT 1 FROM user_favorites WHERE user_id = $1 AND workout_id = $2',
      [userId, workoutId],
    );

    const alreadyFavorited = (exists.rowCount ?? 0) > 0;

    if (alreadyFavorited) {
      await this.db.query(
        'DELETE FROM user_favorites WHERE user_id = $1 AND workout_id = $2',
        [userId, workoutId],
      );
      return { isFavorite: false };
    }

    await this.db.query(
      'INSERT INTO user_favorites (user_id, workout_id) VALUES ($1, $2)',
      [userId, workoutId],
    );
    return { isFavorite: true };
  }
}
