import {
  Controller, Get, Patch, Put,
  Body, Req, UseGuards,
} from '@nestjs/common';
import { AuthGuard } from '../auth/auth.guard';
import { UserService } from './user.service';
import { UpdateProfileDto, UpdateGoalsDto } from './user.dto';

@Controller('user')
@UseGuards(AuthGuard)
export class UserController {
  constructor(private userService: UserService) {}

  @Get()
  async getProfile(@Req() req: any) {
    return this.userService.getProfile(req.userId);
  }

  @Patch('profile')
  async updateProfile(@Req() req: any, @Body() dto: UpdateProfileDto) {
    return this.userService.updateProfile(req.userId, dto);
  }

  @Put('goals')
  async updateGoals(@Req() req: any, @Body() dto: UpdateGoalsDto) {
    return this.userService.updateGoals(req.userId, dto);
  }
}
