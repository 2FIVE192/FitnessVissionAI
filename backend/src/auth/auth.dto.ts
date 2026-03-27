import { IsString, MinLength, IsIn, IsNumber, IsPositive, IsInt, IsOptional } from 'class-validator';

// -- login --

export class LoginDto {
  @IsString()
  @MinLength(1)
  username: string;

  @IsString()
  @MinLength(1)
  password: string;
}

// -- register --

export class RegisterDto {
  @IsString()
  @MinLength(1)
  name: string;

  @IsString()
  @MinLength(3)
  username: string;

  @IsString()
  @MinLength(4)
  password: string;

  @IsIn(['MALE', 'FEMALE'])
  gender: string;

  @IsString()
  dateOfBirth: string;

  @IsNumber()
  @IsPositive()
  weightKg: number;

  @IsNumber()
  @IsPositive()
  heightCm: number;

  @IsInt()
  @IsPositive()
  @IsOptional()
  dailyStepsGoal?: number;

  @IsInt()
  @IsPositive()
  @IsOptional()
  dailyCaloriesGoal?: number;
}
