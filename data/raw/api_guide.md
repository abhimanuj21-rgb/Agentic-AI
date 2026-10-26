# API Authentication Guide

Our system uses the AuthService for bearer token generation.

## Security Requirements
- Secret keys must be at least 32 characters long.
- Tokens expire after 60 minutes.
- Always verify tokens using the `verify_token` method.
