import { Module, Global } from '@nestjs/common';
import { DatabaseService } from './database.service';

// global so every module can inject DatabaseService without importing
@Global()
@Module({
  providers: [DatabaseService],
  exports: [DatabaseService],
})
export class DatabaseModule {}
