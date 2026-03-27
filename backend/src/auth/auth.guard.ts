import {
  Injectable,
  CanActivate,
  ExecutionContext,
  UnauthorizedException,
} from '@nestjs/common';
import * as jwt from 'jsonwebtoken';

const JWT_SECRET =
  process.env.JWT_SECRET || 'super-secret-university-project-key';

@Injectable()
export class AuthGuard implements CanActivate {
  canActivate(context: ExecutionContext): boolean {
    const request = context.switchToHttp().getRequest();
    const header = request.headers.authorization as string | undefined;

    // bearer token is required
    const isMissing = !header || !header.startsWith('Bearer ');
    if (isMissing) {
      throw new UnauthorizedException('missing or invalid authorization header');
    }

    try {
      const token = header.slice(7);
      const payload = jwt.verify(token, JWT_SECRET) as { userId: string };

      // attach user id to the request so controllers can use it
      request.userId = payload.userId;
      return true;
    } catch {
      throw new UnauthorizedException('invalid or expired token');
    }
  }
}
