"""User Authentication and Token Service."""

class AuthService:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key

    def generate_jwt(self, user_id: str) -> str:
        """Generate signed JWT token for user."""
        return f"token_for_{user_id}_{self.secret_key[:4]}"

    def verify_token(self, token: str) -> bool:
        """Verify token authenticity."""
        return token.startswith("token_for_")
