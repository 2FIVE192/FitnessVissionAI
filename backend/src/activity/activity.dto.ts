import { IsString, IsInt, IsNumber, IsOptional, Min, Max } from 'class-validator';

// -- save steps from step counter --

export class SaveStepsDto {
  @IsInt() @Min(0)
  totalStepsSinceBoot: number;

  @IsString()
  timestamp: string;
}

// -- manual activity entry (steps + calories added on top of the day's total) --

export class ManualActivityDto {
  // 0–50 000 steps per manual entry
  @IsOptional() @IsInt() @Min(0) @Max(50000)
  steps?: number;

  // 0–5 000 kcal per manual entry
  @IsNumber() @Min(0) @Max(5000)
  caloriesBurned: number;
}
