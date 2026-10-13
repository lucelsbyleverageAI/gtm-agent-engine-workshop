import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import data_service
from gtm_agent.gtm_agent import (
    build_prospect_profile,
    score_prospect,
    update_prospect_info as update_prospect_info_tool,
)


class ProspectTechnologyUpdateTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "TEST-TECH-UPDATE"
        self.original_record = data_service.PROSPECTS.get(self.prospect_id)
        data_service.PROSPECTS[self.prospect_id] = {
            "name": "Test Prospect",
            "email": "test@example.com",
            "engagement_history": [],
            "account_details": [],
            "tech_stack": ["Python"],
            "annual_revenue": 1_000_000,
        }
        self.original_profile = data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        if self.original_record is None:
            data_service.PROSPECTS.pop(self.prospect_id, None)
        else:
            data_service.PROSPECTS[self.prospect_id] = self.original_record
        data_service._PROFILES.pop(self.prospect_id, None)
        if self.original_profile is not None:
            data_service._PROFILES[self.prospect_id] = self.original_profile

    def test_update_persists_technology_for_fetch(self):
        result = data_service.update_prospect_info(self.prospect_id, "Terraform")

        self.assertTrue(result["updated"])
        self.assertEqual(result["tech_stack"], ["Python", "Terraform"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["Python", "Terraform"])

    def test_update_refreshes_cached_profile(self):
        build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        data_service.update_prospect_info(self.prospect_id, "Terraform")
        updated_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]

        self.assertEqual(updated_profile["tech_stack"], ["Python", "Terraform"])

    def test_scoring_uses_updated_persisted_technology(self):
        profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})["prospect_profile"]
        data_service.update_prospect_info(self.prospect_id, "Terraform")
        llm_result = Mock()
        llm_result.model_dump.return_value = {"score": 80}
        offering = {
            "required_tech_stack": ["Terraform"],
            "min_annual_revenue": 100_000,
            "description": "Infrastructure tooling",
        }

        with patch("gtm_agent.gtm_agent._scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = llm_result
            score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        submitted_profile = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertIn('"Terraform"', submitted_profile)

    def test_unknown_prospect_is_not_reported_as_updated(self):
        result = update_prospect_info_tool.invoke({
            "prospect_id": "UNKNOWN-TECH-UPDATE",
            "technology": "Terraform",
        })

        self.assertFalse(result["updated"])
        self.assertFalse(result["found"])
        self.assertNotIn("tech_stack", result)

    def test_existing_technology_is_not_duplicated(self):
        result = data_service.update_prospect_info(self.prospect_id, "Python")

        self.assertTrue(result["updated"])
        self.assertEqual(result["tech_stack"], ["Python"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["Python"])


if __name__ == "__main__":
    unittest.main()
