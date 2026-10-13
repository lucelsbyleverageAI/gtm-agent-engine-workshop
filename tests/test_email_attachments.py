import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailAttachmentTests(unittest.TestCase):
    def send_email(self, body):
        return send_prospect_email.func(
            prospect={"name": "Avery", "email": "avery@example.com"},
            subject="Pricing information",
            body=body,
            runtime=SimpleNamespace(config={}),
            from_rep={"name": "Morgan", "email": "morgan@example.com"},
        )

    def test_pricing_deck_request_prompt_discloses_limitation(self):
        self.assertIn("cannot include files or collateral", SYSTEM_PROMPT)
        self.assertIn("tell the rep it cannot be attached", SYSTEM_PROMPT)
        self.assertIn("do not claim or imply in the email body", SYSTEM_PROMPT)
        self.assertIn("Do not claim the email or document was sent", SYSTEM_PROMPT)

    def test_attachment_claims_fail_without_sending(self):
        for body in (
            "I've attached the pricing deck.",
            "Please see the attachment.",
            "I included the pricing deck.",
        ):
            with self.subTest(body=body):
                result = self.send_email(body)
                self.assertEqual(result["status"], "failed")
                self.assertIn("Attachments are unsupported", result["error"])
                self.assertNotIn("message_id", result)

    def test_ordinary_email_sends(self):
        result = self.send_email("Would next Tuesday work for a pricing review?")
        self.assertEqual(result["status"], "sent")
        self.assertIn("message_id", result)


if __name__ == "__main__":
    unittest.main()
