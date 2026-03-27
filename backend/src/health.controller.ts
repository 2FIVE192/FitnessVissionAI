import { Controller, Get } from '@nestjs/common';

@Controller('health')
export class HealthController {
  @Get()
  check() {
    const now = new Date().toISOString();
    return { status: 'ok', timestamp: now };
  }
}
