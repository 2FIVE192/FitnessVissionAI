import { IsString, IsOptional, IsUUID } from 'class-validator';

// -- start a workout session --

export class StartSessionDto {
  @IsUUID()
  workoutId: string;
}

// -- complete a session --

export class CompleteSessionDto {
  @IsOptional() @IsString()
  finishedAt?: string;
}
