import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langsmith.run_helpers import tracing_context

from gtm_agent import gtm_agent as agent_module


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def make_agent(tool_calls):
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": name, "args": args, "id": f"call-{index}", "type": "tool_call"}
                for index, (name, args) in enumerate(tool_calls)
            ],
        ),
        AIMessage(content="Workflow complete."),
    ]
    return create_agent(
        model=ToolCallingFakeModel(responses=responses),
        tools=[
            agent_module.lookup_offering,
            agent_module.build_prospect_profile,
            agent_module.get_prospect,
            agent_module.send_prospect_email,
            agent_module.score_prospect,
            agent_module.update_prospect_info,
            agent_module.get_current_rep,
        ],
        system_prompt=agent_module.SYSTEM_PROMPT,
    )


class GTMAgentTests(unittest.TestCase):
    def test_agent_contains_only_the_seven_gtm_tools(self):
        graph = agent_module.gtm_agent.get_graph()
        self.assertEqual(set(graph.nodes), {"__start__", "model", "tools", "__end__"})
        model_function = graph.nodes["model"].data.func
        system_messages = [
            cell.cell_contents
            for cell in model_function.__closure__
            if isinstance(cell.cell_contents, SystemMessage)
        ]
        self.assertEqual(
            [message.content for message in system_messages],
            [agent_module.SYSTEM_PROMPT],
        )
        self.assertEqual(
            set(graph.nodes["tools"].data.tools_by_name),
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )

    def test_get_current_rep_receives_runtime_context(self):
        agent = make_agent([("get_current_rep", {})])
        with tracing_context(enabled=False), patch.object(
            agent_module.data_service,
            "get_rep",
            return_value={"name": "Rep", "email": "rep@example.com"},
        ) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Who am I?"}]},
                config={"metadata": {"user_id": "rep-1"}},
            )
        get_rep.assert_called_once_with("rep-1")
        self.assertEqual(result["messages"][-2].content, '{"rep": {"name": "Rep", "email": "rep@example.com"}, "found": true}')

    def test_send_prospect_email_receives_runtime_context(self):
        agent = make_agent([("send_prospect_email", {
            "prospect": {"name": "Pat", "email": "pat@example.com"},
            "subject": "Hello",
            "body": "Following up.",
        })])
        with tracing_context(enabled=False), patch.object(
            agent_module.data_service,
            "get_rep",
            return_value={"name": "Rep", "email": "rep@example.com"},
        ) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Email Pat."}]},
                config={"metadata": {"user_id": "rep-1"}},
            )
        get_rep.assert_called_once_with("rep-1")
        tool_result = result["messages"][-2].content
        self.assertIn('"status": "sent"', tool_result)
        self.assertIn('"from": "rep@example.com"', tool_result)
        self.assertIn('"to": "pat@example.com"', tool_result)

    def test_run_agent_forwards_run_configuration(self):
        run_id = agent_module.uuid.UUID("00000000-0000-0000-0000-000000000001")
        result_message = AIMessage(content="Done.")
        invoke_result = {"messages": [result_message]}
        with patch.object(agent_module.uuid, "uuid4", return_value=run_id), patch.object(
            agent_module.gtm_agent, "invoke", return_value=invoke_result
        ) as invoke:
            result = agent_module.run_agent(
                "Score prospect LEAD-1",
                user_id="rep-1",
                environment="staging",
                thread_id="thread-1",
            )
        self.assertEqual(result["run_id"], str(run_id))
        self.assertEqual(result["reply"], "Done.")
        invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "Score prospect LEAD-1"}]},
            config={
                "run_name": "GTM Assistant",
                "run_id": run_id,
                "metadata": {
                    "thread_id": "thread-1",
                    "user_id": "rep-1",
                    "environment": "staging",
                    "request_intent": "score_prospect",
                },
            },
        )

    def test_lookup_score_and_email_workflow_uses_gtm_tools(self):
        offering = {
            "offering_id": "OFFER-1",
            "required_tech_stack": ["CRM"],
            "min_annual_revenue": 1000000,
            "description": "CRM solution",
        }
        profile = {"prospect_id": "LEAD-1", "name": "Pat", "tech_stack": ["CRM"]}
        prospect = {"prospect_id": "LEAD-1", "name": "Pat", "email": "pat@example.com"}
        tool_calls = [
            ("get_current_rep", {}),
            ("lookup_offering", {"offering_id": "OFFER-1"}),
            ("build_prospect_profile", {"prospect_id": "LEAD-1"}),
            ("score_prospect", {"prospect_profile": profile, "offering": offering}),
            ("get_prospect", {"prospect_id": "LEAD-1"}),
            ("send_prospect_email", {
                "prospect": prospect,
                "subject": "A good fit",
                "body": "Let's talk.",
            }),
        ]
        agent = make_agent(tool_calls)
        score = agent_module.ProspectScore(
            score=90,
            justification="Strong fit.",
            rubric_breakdown={"revenue_fit": 30, "tech_stack_match": 40, "segment_fit": 20},
        )
        with (
            tracing_context(enabled=False),
            patch.object(agent_module.data_service, "get_rep", return_value={"name": "Rep", "email": "rep@example.com"}),
            patch.object(agent_module.data_service, "get_offering", return_value=offering),
            patch.object(agent_module.data_service, "get_profile_from_db", return_value={"prospect_profile": profile}),
            patch.object(agent_module.data_service, "get_prospect_record", return_value=prospect),
            patch.object(agent_module.data_service, "fetch_tech_stack", return_value=["CRM"]),
            patch.object(agent_module, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = score
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Score and email Pat."}]},
                config={"metadata": {"user_id": "rep-1"}},
            )
        tool_results = [message.content for message in result["messages"] if message.type == "tool"]
        self.assertEqual(len(tool_results), len(tool_calls))
        self.assertTrue(any('"score": 90' in content for content in tool_results))
        self.assertTrue(any('"status": "sent"' in content for content in tool_results))
        self.assertEqual(result["messages"][-1].content, "Workflow complete.")


if __name__ == "__main__":
    unittest.main()
