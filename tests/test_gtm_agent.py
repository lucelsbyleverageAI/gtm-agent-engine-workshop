import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import gtm_agent as agent_module
from gtm_agent import data_service


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep-1"}})
        self.prospect = {
            "prospect_id": "LEAD-12345",
            "name": "Alex Example",
            "email": "alex@example.com",
        }
        self.from_rep = {"name": "Rep Example", "email": "rep@example.com"}

    def send_email(self, prospect=None):
        return agent_module.send_prospect_email.func(
            prospect=prospect or self.prospect,
            subject="Hello",
            body="Checking in",
            runtime=self.runtime,
            from_rep=self.from_rep,
        )

    def test_disqualified_prospect_is_blocked_without_creating_message_id(self):
        with (
            patch.object(data_service, "get_prospect_record", return_value={"disqualified": True}) as get_record,
            patch.object(agent_module.uuid, "uuid4") as create_message_id,
        ):
            result = self.send_email()

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is marked disqualified; email not sent."},
        )
        self.assertNotIn("message_id", result)
        get_record.assert_called_once_with("LEAD-12345")
        create_message_id.assert_not_called()

    def test_eligible_prospect_is_sent(self):
        with (
            patch.object(data_service, "get_prospect_record", return_value={"disqualified": False}),
            patch.object(agent_module.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)),
        ):
            result = self.send_email()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["message_id"], "msg-" + "a" * 12)

    def test_missing_source_record_preserves_send_behavior(self):
        with (
            patch.object(data_service, "get_prospect_record", return_value=None),
            patch.object(agent_module.uuid, "uuid4", return_value=SimpleNamespace(hex="b" * 32)),
        ):
            result = self.send_email()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["message_id"], "msg-" + "b" * 12)

    def test_missing_email_preserves_existing_failure(self):
        prospect = {"prospect_id": "LEAD-12345", "name": "Alex Example"}
        with patch.object(data_service, "get_prospect_record") as get_record:
            result = self.send_email(prospect)

        self.assertEqual(
            result,
            {"status": "failed", "error": "Prospect record has no email address."},
        )
        get_record.assert_not_called()

    def test_blocked_sends_are_explained_without_retry(self):
        self.assertIn("If a send is blocked, do not retry", agent_module.SYSTEM_PROMPT)
        self.assertIn("tell the rep that the email was not sent and explain why", agent_module.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
