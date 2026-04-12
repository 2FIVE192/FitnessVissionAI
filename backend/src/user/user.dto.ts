import {
  IsString, IsOptional, MinLength, MaxLength, IsIn,
  IsNumber, IsInt, Min, Max,
} from 'class-validator';

// -- update profile (partial) --

export class UpdateProfileDto {
  @IsOptional() @IsIn(['MALE', 'FEMALE'])
  gender?: string;

  @IsOptional() @IsString()
  dateOfBirth?: string;

  // 20–300 kg
  @IsOptional() @IsNumber() @Min(20) @Max(300)
  weightKg?: number;

  // 100–250 cm
  @IsOptional() @IsNumber() @Min(100) @Max(250)
  heightCm?: number;

  @IsOptional() @IsString() @MinLength(3) @MaxLength(50)
  username?: string;

  @IsOptional() @IsString() @MinLength(4) @MaxLength(100)
  password?: string;
}

// -- update goals --

export class UpdateGoalsDto {
  // 500–50 000 steps
  @IsInt() @Min(500) @Max(50000)
  stepsGoal: number;

  // 50–5 000 kcal
  @IsInt() @Min(50) @Max(5000)
  caloriesGoal: number;
}
