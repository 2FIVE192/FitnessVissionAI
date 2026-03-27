import {
  Controller, Post, Put, Body, Param, Req, UseGuards, HttpCode,
} from '@nestjs/common';
import { AuthGuard } from '../auth/auth.guard';
import { SessionsService } from './sessions.service';
import { StartSessionDto, CompleteSessionDto } from './sessions.dto';

@Controller('sessions')
@UseGuards(AuthGuard)
export class SessionsController {
  constructor(private sessionsService: SessionsService) {}

  @Post('start')
  @HttpCode(201)
  async start(@Req() req: any, @Body() dto: StartSessionDto) {
    return this.sessionsService.startSession(req.userId, dto.workoutId);
  }

  @Put(':id/complete')
  async complete(
    @Req() req: any,
    @Param('id') id: string,
    @Body() dto: CompleteSessionDto,
  ) {
    return this.sessionsService.completeSession(req.userId, id, dto.finishedAt);
  }

  @Put(':id/abandon')
  async abandon(@Req() req: any, @Param('id') id: string) {
    return this.sessionsService.abandonSession(req.userId, id);
  }
}
