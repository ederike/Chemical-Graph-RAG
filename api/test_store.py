"""Account isolation tests. Run: CGR_WEB_DB=/tmp/cgr-test.db python -m api.test_store"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class StoreIsolation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["CGR_WEB_DB"] = str(Path(self.tmp.name) / "app.db")
        from api import store

        store.init_db()
        self.store = store

    def tearDown(self):
        self.tmp.cleanup()

    def test_seed_admin_and_extra_user(self):
        users = {u["username"] for u in self.store.list_users()}
        self.assertIn("admin", users)
        self.store.create_user("alice", "alice-secret")
        names = [u["username"] for u in self.store.list_users()]
        self.assertIn("alice", names)

    def test_wrong_password(self):
        self.assertIsNone(self.store.authenticate("admin", "nope"))
        self.assertIsNotNone(self.store.authenticate("admin", "kaiyin"))

    def test_conversations_are_private(self):
        alice = self.store.create_user("alice", "pw-a")
        bob = self.store.create_user("bob", "pw-b")
        conv = self.store.create_conversation(alice["id"], "Alice 的配方")
        self.store.add_turn(
            alice["id"],
            conv["id"],
            query="相对密度？",
            mode="agentic",
            answer="1.10",
            status=1,
        )
        self.assertIsNone(self.store.get_conversation(bob["id"], conv["id"]))
        self.assertFalse(self.store.delete_conversation(bob["id"], conv["id"]))
        self.assertEqual(self.store.list_conversations(bob["id"]), [])
        mine = self.store.get_conversation(alice["id"], conv["id"])
        self.assertIsNotNone(mine)
        self.assertEqual(len(mine["turns"]), 1)
        self.assertEqual(mine["turns"][0]["query"], "相对密度？")
        listed = self.store.list_conversations(alice["id"])
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], conv["id"])

    def test_token_roundtrip(self):
        user = self.store.authenticate("admin", "kaiyin")
        token = self.store.issue_token(user["id"])
        got = self.store.user_from_token(token)
        self.assertEqual(got["username"], "admin")
        self.store.revoke_token(token)
        self.assertIsNone(self.store.user_from_token(token))

    def test_expired_and_malformed_token(self):
        user = self.store.authenticate("admin", "kaiyin")
        token = self.store.issue_token(user["id"])
        conn = self.store.connect()
        try:
            conn.execute(
                "UPDATE auth_tokens SET expires_at = ? WHERE token_hash = ?",
                ("2000-01-01T00:00:00Z", self.store.hash_token(token)),
            )
            conn.commit()
        finally:
            conn.close()
        self.assertIsNone(self.store.user_from_token(token))

        token2 = self.store.issue_token(user["id"])
        conn = self.store.connect()
        try:
            conn.execute(
                "UPDATE auth_tokens SET expires_at = ? WHERE token_hash = ?",
                ("not-a-date", self.store.hash_token(token2)),
            )
            conn.commit()
        finally:
            conn.close()
        self.assertIsNone(self.store.user_from_token(token2))

    def test_like_wildcards_are_literal(self):
        alice = self.store.create_user("alice", "pw-a")
        self.store.create_conversation(alice["id"], "hello50world")
        self.store.create_conversation(alice["id"], "base 50% PVC")
        hit = self.store.list_conversations(alice["id"], q="50%")
        self.assertEqual(len(hit), 1)
        self.assertEqual(hit[0]["title"], "base 50% PVC")
        miss = self.store.list_conversations(alice["id"], q="hello_")
        self.assertEqual(miss, [])

    def test_password_too_long(self):
        long_pw = "x" * (self.store.MAX_PASSWORD_LEN + 1)
        with self.assertRaises(ValueError):
            self.store.create_user("bob", long_pw)
        self.assertIsNone(self.store.authenticate("admin", long_pw))


if __name__ == "__main__":
    unittest.main()
