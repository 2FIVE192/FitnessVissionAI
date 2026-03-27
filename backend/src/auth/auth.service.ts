import {
  Injectable,
  ConflictException,
  UnauthorizedException,
} from '@nestjs/common';
import * as bcrypt from 'bcrypt';
import * as jwt from 'jsonwebtoken';
import { DatabaseService } from '../database/database.service';
import { RegisterDto, LoginDto } from './auth.dto';

const JWT_SECRET =
  process.env.JWT_SECRET || 'super-secret-university-project-key';
const SALT_ROUNDS = 10;
const TOKEN_EXPIRY = '30d';

@Injectable()
export class AuthService {
  constructor(private db: DatabaseService) {}

  // sign a jwt for the given user id
  private signToken(userId: string): string {
    return jwt.sign({ userId }, JWT_SECRET, { expiresIn: TOKEN_EXPIRY });
  }

  async register(dto: RegisterDto) {
    const {
      name, username, password, gender, dateOfBirth,
      weightKg, heightCm,
    } = dto;

    const stepsGoal = dto.dailyStepsGoal ?? 10000;
    const caloriesGoal = dto.dailyCaloriesGoal ?? 500;

    // check if username is already taken
    const existing = await this.db.query(
      'SELECT 1 FROM users WHERE username = $1',
      [username],
    );
    const alreadyTaken = (existing.rowCount ?? 0) > 0;

    if (alreadyTaken) {
      throw new ConflictException('username already taken');
    }

    const passwordHash = await bcrypt.hash(password, SALT_ROUNDS);

    const result = await this.db.query(
      `INSERT INTO users (name, username, password_hash, gender, date_of_birth,
                          weight_kg, height_cm, daily_steps_goal, daily_calories_goal)
       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *`,
      [name, username, passwordHash, gender, dateOfBirth,
        weightKg, heightCm, stepsGoal, caloriesGoal],
    );

    const user = result.rows[0];
    const token = this.signToken(user.id);

    return { token, user: this.formatUser(user) };
  }

  async login(dto: LoginDto) {
    const { username, password } = dto;

    const result = await this.db.query(
      'SELECT * FROM users WHERE username = $1',
      [username],
    );
    const userNotFound = !result.rowCount || result.rowCount === 0;

    if (userNotFound) {
      throw new UnauthorizedException('invalid credentials');
    }

    const user = result.rows[0];
    const passwordValid = await bcrypt.compare(password, user.password_hash);

    if (!passwordValid) {
      throw new UnauthorizedException('invalid credentials');
    }

    const token = this.signToken(user.id);
    return { token, user: this.formatUser(user) };
  }

  // maps a raw db row into a clean user response object
  formatUser(row: any) {
    return {
      id: row.id,
      name: row.name,
      username: row.username,
      gender: row.gender,
      dateOfBirth: row.date_of_birth,
      weightKg: row.weight_kg,
      heightCm: row.height_cm,
      dailyStepsGoal: row.daily_steps_goal,
      dailyCaloriesGoal: row.daily_calories_goal,
    };
  }
}
