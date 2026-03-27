import { Injectable, OnModuleDestroy } from '@nestjs/common';
import { Pool, QueryResult } from 'pg';

@Injectable()
export class DatabaseService implements OnModuleDestroy {
  private pool: Pool;

  constructor() {
    // connect using env var or fall back to local docker-compose defaults
    const connectionString =
      process.env.DATABASE_URL ||
      'postgresql://fitness:fitness@localhost:5435/fitness_db';

    this.pool = new Pool({ connectionString });
  }

  // run a parameterized query against the connection pool
  async query(text: string, params?: any[]): Promise<QueryResult> {
    return this.pool.query(text, params);
  }

  async onModuleDestroy() {
    await this.pool.end();
  }
}
