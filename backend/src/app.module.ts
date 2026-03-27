import { Module } from '@nestjs/common';
import { DatabaseModule } from './database/database.module';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './user/user.module';
import { ActivityModule } from './activity/activity.module';
import { WorkoutsModule } from './workouts/workouts.module';
import { SessionsModule } from './sessions/sessions.module';
import { HealthController } from './health.controller';

@Module({
  imports: [
    DatabaseModule,
    AuthModule,
    UserModule,
    ActivityModule,
    WorkoutsModule,
    SessionsModule,
  ],
  controllers: [HealthController],
})
export class AppModule {}
