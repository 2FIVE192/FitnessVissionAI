import { IsString, IsInt, IsNumber, IsOptional, Min } from 'class-validator';

// -- save steps from step counter --

export class SaveStepsDto {
  @IsInt() @Min(0)
  totalStepsSinceBoot: number;

  @IsString()
  timestamp: string;
}

// -- manual activity entry (steps + calories added on top of the day's total) --

export class ManualActivityDto {
  @IsOptional() @IsInt() @Min(0)
  steps?: number;

  @IsNumber() @Min(0)
  caloriesBurned: number;
}
