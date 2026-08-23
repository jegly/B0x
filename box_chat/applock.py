"""App Lock — an Argon2id-hashed password gate on the app window.

Ported from Jegly's Tesseract app (tesseract-gui applock.rs). This is NOT
encryption: Box has no encrypted volumes or files, and the hash is not a
key. It only gates casual access to the app UI — a padlock on the door, not
a safe. No padlock iconography by design (matches Tesseract's stated reason:
it advertises "there is something to unlock here" to a shoulder-surfer).

The hash lives in ``settings.app_lock_hash``; ``Settings.save`` chmods the
settings file to 0600 so no other local account can read it and brute-force
it offline. Engine-tier / gi-free — the window drives the lock UI.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__all__ = ["hash_password", "verify_password", "PasswordHasher"]

# Argon2id parameters — interactive defaults from argon2-cffi, ample for a
# local UI gate (this isn't protecting a key, just the app window).
_TIME_COST = 3
_MEMORY_COST = 64 * 1024  # 64 MiB
_PARALLELISM = 4


def _hasher():
    from argon2 import PasswordHasher as _PH

    return _PH(
        time_cost=_TIME_COST,
        memory_cost=_MEMORY_COST,
        parallelism=_PARALLELISM,
    )


def hash_password(password: str) -> str:
    """Return an Argon2id encoded hash string for ``password``."""
    return _hasher().hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """True iff ``password`` matches ``stored_hash``. Never raises on a
    mismatch — a wrong password is a normal outcome, not an error."""
    if not stored_hash:
        return False
    from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

    try:
        return _hasher().verify(stored_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False
    except Exception:  # noqa: BLE001 — any hashing error = not authenticated
        log.exception("app-lock verify failed")
        return False


# Alias so callers can hold a single object if preferred.
class PasswordHasher:
    @staticmethod
    def hash(password: str) -> str:
        return hash_password(password)

    @staticmethod
    def verify(stored_hash: str, password: str) -> bool:
        return verify_password(stored_hash, password)
