import {
  Controller, Get, Post, Body, Req, UseGuards, Query, ParseIntPipe, DefaultValuePipe,
} from '@nestjs/common';
import { AuthGuard } from '../auth/auth.guard';
import { ActivityService } from './activity.service';
import { SaveStepsDto, ManualActivityDto } from './activity.dto';

@Controller('activity')
@UseGuards(AuthGuard)
export class ActivityController {
  constructor(private activityService: ActivityService) {}

  @Get('today')
  async getToday(@Req() req: any) {
    return this.activityService.getTodayActivity(req.userId);
  }

  @Get('history')
  async getHistory(
    @Req() req: any,
    @Query('days', new DefaultValuePipe(7), ParseIntPipe) days: number,
  ) {
    const clampedDays = Math.min(Math.max(days, 1), 90);
    return this.activityService.getActivityHistory(req.userId, clampedDays);
  }

  // called by the step counter worker to sync steps to backend
  @Post('steps')
  async saveSteps(@Req() req: any, @Body() dto: SaveStepsDto) {
    return this.activityService.saveSteps(
      req.userId,
      dto.totalStepsSinceBoot,
      dto.timestamp,
    );
  }

  // called when the user manually logs steps and/or calories
  @Post('manual')
  async saveManual(@Req() req: any, @Body() dto: ManualActivityDto) {
    return this.activityService.saveManualActivity(
      req.userId,
      dto.steps ?? 0,
      dto.caloriesBurned,
    );
  }
}
