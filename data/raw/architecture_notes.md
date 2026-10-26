# System Architecture & API Design Notes

## 1. Authentication Layer
All incoming microservice requests must bear an `Authorization: Bearer <token>` header.
Tokens are verified by the `TokenService` using HMAC-SHA256 signatures with 60-minute TTLs.

### Token Expiration Policy
- Access tokens expire in 1 hour (3600 seconds).
- Revoked tokens are kept in an in-memory TTL blacklist cache.
- Role-based permissions supported: `developer`, `admin`, `auditor`.

## 2. Database Connection Pooling
- PostgreSQL connection pool size is capped at 20 concurrent connections per worker.
- Maximum idle connection duration is 300 seconds.
- SSL connection mode: `verify-full` in production environments.

## 3. Error Handling Standard
All APIs return structured JSON errors with:
```json
{
  "error_code": "INVALID_CREDENTIALS",
  "message": "Human-readable explanation",
  "status_code": 401
}
```
