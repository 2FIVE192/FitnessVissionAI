import {
  IsString, MinLength, MaxLength, IsIn, IsNumber, IsPositive,
  IsInt, IsOptional, Min, Max, IsDateString,
} from 'class-validator';

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
  @MaxLength(100)
  name: string;

  @IsString()
  @MinLength(3)
  @MaxLength(50)
  username: string;

  @IsString()
  @MinLength(4)
  @MaxLength(100)
  password: string;

  @IsIn(['MALE', 'FEMALE'])
  gender: string;

  @IsDateString()
  dateOfBirth: string;

  // realistic human weight: 20–300 kg
  @IsNumber()
  @Min(20)
  @Max(300)
  weightKg: number;

  // realistic human height: 100–250 cm
  @IsNumber()
  @Min(100)
  @Max(250)
  heightCm: number;

  // daily steps goal: 500–50 000
  @IsInt()
  @Min(500)
  @Max(50000)
  @IsOptional()
  dailyStepsGoal?: number;

  // daily calories goal: 50–5 000 kcal
  @IsInt()
  @Min(50)
  @Max(5000)
  @IsOptional()
  dailyCaloriesGoal?: number;
}
