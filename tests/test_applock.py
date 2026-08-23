"""App Lock — Argon2id hash/verify roundtrip and failure modes.

Covers box_chat/applock.py: a correct password verifies, a wrong one is a
quiet False (never an exception), an empty/garbage stored hash is False, and
the PasswordHasher facade matches the module functions.
"""
from __future__ import annotations

import unittest

from box_chat import applock


class HashRoundtripTests(unittest.TestCase):
    def test_correct_password_verifies(self) -> None:
        h = applock.hash_password("correct horse battery staple")
        self.assertTrue(applock.verify_password(h, "correct horse battery staple"))

    def test_wrong_password_is_false_not_raise(self) -> None:
        h = applock.hash_password("swordfish")
        # A wrong password is a normal outcome — must return False, never raise.
        self.assertFalse(applock.verify_password(h, "swordfish "))
        self.assertFalse(applock.verify_password(h, "Swordfish"))
        self.assertFalse(applock.verify_password(h, ""))

    def test_hash_is_argon2id_and_salted(self) -> None:
        h1 = applock.hash_password("hunter2")
        h2 = applock.hash_password("hunter2")
        self.assertTrue(h1.startswith("$argon2id$"))
        # Distinct salts → distinct encoded hashes for the same password.
        self.assertNotEqual(h1, h2)
        self.assertTrue(applock.verify_password(h1, "hunter2"))
        self.assertTrue(applock.verify_password(h2, "hunter2"))

    def test_empty_stored_hash_is_false(self) -> None:
        self.assertFalse(applock.verify_password("", "anything"))

    def test_garbage_stored_hash_is_false(self) -> None:
        # Malformed hash → InvalidHashError swallowed → False, no crash.
        self.assertFalse(applock.verify_password("not-a-real-hash", "anything"))

    def test_unicode_password(self) -> None:
        h = applock.hash_password("pÃ¡sswörd–🔐")
        self.assertTrue(applock.verify_password(h, "pÃ¡sswörd–🔐"))
        self.assertFalse(applock.verify_password(h, "password"))


class PasswordHasherFacadeTests(unittest.TestCase):
    def test_facade_matches_module_functions(self) -> None:
        h = applock.PasswordHasher.hash("s3cret")
        self.assertTrue(applock.PasswordHasher.verify(h, "s3cret"))
        self.assertFalse(applock.PasswordHasher.verify(h, "wrong"))
        # Cross-check: a module-produced hash verifies via the facade too.
        h2 = applock.hash_password("s3cret")
        self.assertTrue(applock.PasswordHasher.verify(h2, "s3cret"))


if __name__ == "__main__":
    unittest.main()
