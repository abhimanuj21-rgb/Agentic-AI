"""Authentication and Session Management Service."""

import hashlib
import hmac
import time
from typing import Dict, Optional


class TokenService:
    """Handles JWT generation, verification, and revocation."""

    def __init__(self, secret_key: str, token_ttl_seconds: int = 3600):
        self.secret_key = secret_key.encode("utf-8")
        self.token_ttl_seconds = token_ttl_seconds
        self._revocation_store: Dict[str, float] = {}

    def generate_token(self, user_id: str, role: str = "developer") -> str:
        """Create a secure signature-verified token."""
        expires_at = time.time() + self.token_ttl_seconds
        payload = f"{user_id}:{role}:{int(expires_at)}"
        signature = hmac.new(self.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{payload}:{signature}"

    def verify_token(self, token: str) -> Optional[Dict[str, str]]:
        """Verify token expiration, integrity, and revocation status."""
        try:
            user_id, role, exp_str, signature = token.split(":")
            if token in self._revocation_store:
                return None  # Token was revoked

            if time.time() > float(exp_str):
                return None  # Token expired

            payload = f"{user_id}:{role}:{exp_str}"
            expected_sig = hmac.new(self.secret_key, payload.encode("utf-8"), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_sig):
                return None  # Tampered token

            return {"user_id": user_id, "role": role, "expires_at": exp_str}
        except (ValueError, AttributeError):
            return None

    def revoke_token(self, token: str) -> bool:
        """Mark a token as revoked."""
        self._revocation_store[token] = time.time()
        return True
