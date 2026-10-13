import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service

with (
    patch("langchain_openai.ChatOpenAI") as chat_openai,
    patch("deepagents.create_deep_agent", return_value=Mock()),
):
    chat_openai.return_value.with_structured_output.return_value = Mock()
    from gtm_agent import gtm_agent


class ProspectDataSanitizationTests(unittest.TestCase):
    def setUp(self):
        self.profiles = data_service._PROFILES.copy()
        data_service._PROFILES.clear()

    def tearDown(self):
        data_service._PROFILES.clear()
        data_service._PROFILES.update(self.profiles)

    def test_get_prospect_returns_only_allowed_contact_fields(self):
        result = gtm_agent.get_prospect.func("LEAD-12853")

        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": "LEAD-12853",
                "name": "Omar Okafor",
                "email": "omar.okafor@lakesideanalytics.com",
                "disqualified": False,
            },
        )
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_get_prospect_omits_missing_optional_contact_fields(self):
        with patch.object(
            data_service,
            "get_prospect_record",
            return_value={"billing_qualification": {"tax_id": "private"}, "annual_revenue": 42},
        ):
            result = gtm_agent.get_prospect.func("LEAD-TEST")

        self.assertEqual(result["prospect"], {"prospect_id": "LEAD-TEST"})

    def test_fresh_profile_is_sanitized_before_return_and_cache(self):
        record = {
            "name": "Taylor Example",
            "email": "taylor@example.com",
            "annual_revenue": 123,
            "billing_qualification": {"tax_id": "private", "card_on_file": "private"},
            "date_of_birth": "private",
            "enrichment_source": "Apollo",
        }
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value=[{"industry": "Software"}]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
        ):
            result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        expected = {
            "prospect_id": "LEAD-TEST",
            "name": "Taylor Example",
            "email": "taylor@example.com",
            "annual_revenue": 123,
            "enrichment_source": "Apollo",
            "engagement_history": [],
            "account_details": [{"industry": "Software"}],
            "tech_stack": ["Kafka"],
        }
        self.assertEqual(result["prospect_profile"], expected)
        self.assertEqual(data_service._PROFILES["LEAD-TEST"], expected)

    def test_cached_profile_is_sanitized_before_return_and_cache(self):
        data_service._PROFILES["LEAD-TEST"] = {
            "prospect_id": "LEAD-TEST",
            "name": "Taylor Example",
            "account_details": [{"industry": "Software", "payment_method": "private"}],
            "billing_qualification": {"credit_check_ref": "private"},
            "custom_identity_data": "private",
        }

        result = gtm_agent.build_prospect_profile.func("LEAD-TEST")

        expected = {
            "prospect_id": "LEAD-TEST",
            "name": "Taylor Example",
            "account_details": [{"industry": "Software"}],
        }
        self.assertEqual(result["prospect_profile"], expected)
        self.assertEqual(data_service._PROFILES["LEAD-TEST"], expected)

    def test_cache_read_sanitizes_before_returning_from_traced_helper(self):
        data_service._PROFILES["LEAD-TEST"] = {
            "prospect_id": "LEAD-TEST",
            "billing_qualification": {"tax_id": "private"},
        }

        result = data_service.get_profile_from_db("LEAD-TEST")

        self.assertEqual(result, {"prospect_profile": {"prospect_id": "LEAD-TEST"}})
        self.assertEqual(data_service._PROFILES["LEAD-TEST"], {"prospect_id": "LEAD-TEST"})

    def test_scoring_prompt_includes_only_scoring_fields(self):
        profile = {
            "prospect_id": "LEAD-TEST",
            "name": "Taylor Example",
            "email": "taylor@example.com",
            "annual_revenue": 123,
            "tech_stack": ["stale"],
            "account_details": [{"industry": "Software"}],
            "billing_qualification": {"tax_id": "private"},
            "date_of_birth": "private",
            "custom_field": "private",
        }
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 100,
            "description": "A useful offering",
        }
        scoring_result = Mock()
        scoring_result.model_dump.return_value = {"score": 90}
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(gtm_agent._scoring_llm, "invoke", return_value=scoring_result) as invoke,
        ):
            result = gtm_agent.score_prospect.func(profile, offering)

        user_message = invoke.call_args.args[0][1]["content"]
        scoring_profile = json.loads(user_message.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(
            scoring_profile,
            {
                "prospect_id": "LEAD-TEST",
                "name": "Taylor Example",
                "annual_revenue": 123,
                "tech_stack": ["Kafka"],
                "account_details": [{"industry": "Software"}],
            },
        )
        self.assertEqual(result, {"score": 90})


if __name__ == "__main__":
    unittest.main()
