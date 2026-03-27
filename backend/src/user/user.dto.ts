import {
  IsString, IsOptional, MinLength, IsIn,
  IsNumber, IsPositive, IsInt,
} from 'class-validator';

// -- update profile (partial) --

export class UpdateProfileDto {
  @IsOptional() @IsIn(['MALE', 'FEMALE'])
  gender?: string;

  @IsOptional() @IsString()
  dateOfBirth?: string;

  @IsOptional() @IsNumber() @IsPositive()
  weightKg?: number;

  @IsOptional() @IsNumber() @IsPositive()
  heightCm?: number;

  @IsOptional() @IsString() @MinLength(3)
  username?: string;

  @IsOptional() @IsString() @MinLength(4)
  password?: string;
}

// -- update goals --

export class UpdateGoalsDto {
  @IsInt() @IsPositive()
  stepsGoal: number;

  @IsInt() @IsPositive()
  caloriesGoal: number;
}
