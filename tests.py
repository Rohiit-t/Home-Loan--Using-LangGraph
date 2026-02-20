"""Tests for the Home Loan LangGraph agent (no OpenAI API key required)."""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out openai / langchain_openai so we can test without an API key
# ---------------------------------------------------------------------------

def _make_openai_stub():
    """Return a minimal stub module for langchain_openai."""
    stub = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, prompt):
            return MagicMock(content="[DOCUMENTS_COMPLETE]")

    stub.ChatOpenAI = FakeChatOpenAI
    return stub


# Patch before any project imports
sys.modules.setdefault("langchain_openai", _make_openai_stub())
os.environ.setdefault("OPENAI_API_KEY", "test-key")

# Now safe to import project modules
from state import LoanState  # noqa: E402
from graph import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    base = dict(
        messages=[],
        applicant_name=None,
        applicant_age=None,
        employment_status=None,
        annual_income=None,
        credit_score=None,
        existing_debt=None,
        down_payment=None,
        loan_amount=None,
        loan_term_years=None,
        property_value=None,
        property_address=None,
        current_stage="document_collection",
        documents_collected=False,
        eligibility_passed=False,
        credit_check_passed=False,
        property_valuation_passed=False,
        loan_decision=None,
        rejection_reason=None,
        approved_amount=None,
        interest_rate=None,
        monthly_payment=None,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoanStateSchema(unittest.TestCase):
    """LoanState TypedDict has all required fields."""

    def test_required_fields_present(self):
        state = _base_state()
        for field in (
            "messages",
            "applicant_name",
            "annual_income",
            "credit_score",
            "loan_amount",
            "property_value",
            "current_stage",
            "documents_collected",
            "eligibility_passed",
            "credit_check_passed",
            "property_valuation_passed",
            "loan_decision",
        ):
            self.assertIn(field, state)

    def test_default_flags_are_false(self):
        state = _base_state()
        self.assertFalse(state["documents_collected"])
        self.assertFalse(state["eligibility_passed"])
        self.assertFalse(state["credit_check_passed"])
        self.assertFalse(state["property_valuation_passed"])


class TestGraphStructure(unittest.TestCase):
    """LangGraph compiles without errors and has the expected nodes."""

    def test_graph_compiles(self):
        graph = build_graph()
        self.assertIsNotNone(graph)

    def test_graph_has_all_nodes(self):
        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {
            "document_collection",
            "eligibility_check",
            "credit_check",
            "property_valuation",
            "underwriting",
            "loan_approval",
            "loan_rejection",
        }
        self.assertTrue(expected.issubset(node_names), f"Missing nodes: {expected - node_names}")


class TestRoutingFunctions(unittest.TestCase):
    """Conditional edge routing functions behave correctly."""

    def setUp(self):
        # Import routing functions directly
        import graph as g
        self.route_docs = g.route_after_documents
        self.route_eligibility = g.route_after_eligibility
        self.route_credit = g.route_after_credit
        self.route_valuation = g.route_after_valuation
        self.route_underwriting = g.route_after_underwriting

    def test_route_after_documents_incomplete(self):
        state = _base_state(documents_collected=False)
        self.assertEqual(self.route_docs(state), "document_collection")

    def test_route_after_documents_complete(self):
        state = _base_state(documents_collected=True)
        self.assertEqual(self.route_docs(state), "eligibility_check")

    def test_route_after_eligibility_pass(self):
        state = _base_state(eligibility_passed=True)
        self.assertEqual(self.route_eligibility(state), "credit_check")

    def test_route_after_eligibility_fail(self):
        state = _base_state(eligibility_passed=False)
        self.assertEqual(self.route_eligibility(state), "loan_rejection")

    def test_route_after_credit_pass(self):
        state = _base_state(credit_check_passed=True)
        self.assertEqual(self.route_credit(state), "property_valuation")

    def test_route_after_credit_fail(self):
        state = _base_state(credit_check_passed=False)
        self.assertEqual(self.route_credit(state), "loan_rejection")

    def test_route_after_valuation_pass(self):
        state = _base_state(property_valuation_passed=True)
        self.assertEqual(self.route_valuation(state), "underwriting")

    def test_route_after_valuation_fail(self):
        state = _base_state(property_valuation_passed=False)
        self.assertEqual(self.route_valuation(state), "loan_rejection")

    def test_route_after_underwriting_approved(self):
        state = _base_state(loan_decision="approved")
        self.assertEqual(self.route_underwriting(state), "loan_approval")

    def test_route_after_underwriting_rejected(self):
        state = _base_state(loan_decision="rejected")
        self.assertEqual(self.route_underwriting(state), "loan_rejection")


class TestNodeLogic(unittest.TestCase):
    """Unit-test individual node functions with mocked LLM responses."""

    def _patch_llm(self, content: str):
        """Return a context manager that patches the shared _llm in nodes."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=content)
        return patch("nodes._llm", mock_llm)

    def test_document_collection_marks_complete(self):
        from nodes import document_collection_node
        state = _base_state()
        with self._patch_llm("All done! [DOCUMENTS_COMPLETE]"):
            result = document_collection_node(state)
        self.assertTrue(result.get("documents_collected"))
        self.assertEqual(result.get("current_stage"), "eligibility_check")

    def test_document_collection_continues(self):
        from nodes import document_collection_node
        state = _base_state()
        with self._patch_llm("What is your name?"):
            result = document_collection_node(state)
        self.assertFalse(result.get("documents_collected", False))

    def test_eligibility_check_pass(self):
        from nodes import eligibility_check_node
        state = _base_state(applicant_age=30, annual_income=80000, employment_status="employed")
        with self._patch_llm("Looks good. [ELIGIBLE]"):
            result = eligibility_check_node(state)
        self.assertTrue(result.get("eligibility_passed"))
        self.assertEqual(result.get("current_stage"), "credit_check")

    def test_eligibility_check_fail(self):
        from nodes import eligibility_check_node
        state = _base_state(applicant_age=16, annual_income=10000)
        with self._patch_llm("Too young. [NOT_ELIGIBLE: Applicant must be at least 18]"):
            result = eligibility_check_node(state)
        self.assertFalse(result.get("eligibility_passed"))
        self.assertEqual(result.get("loan_decision"), "rejected")
        self.assertIn("18", result.get("rejection_reason", ""))

    def test_credit_check_pass(self):
        from nodes import credit_check_node
        state = _base_state(credit_score=720, annual_income=90000, existing_debt=500)
        with self._patch_llm("Strong credit. [CREDIT_APPROVED]"):
            result = credit_check_node(state)
        self.assertTrue(result.get("credit_check_passed"))
        self.assertEqual(result.get("current_stage"), "property_valuation")

    def test_credit_check_missing_income(self):
        from nodes import credit_check_node
        state = _base_state(credit_score=720, annual_income=None, existing_debt=500)
        # No LLM patch needed — should short-circuit before calling the LLM
        result = credit_check_node(state)
        self.assertFalse(result.get("credit_check_passed"))
        self.assertEqual(result.get("loan_decision"), "rejected")
        self.assertEqual(result.get("current_stage"), "rejection")

    def test_credit_check_fail(self):
        from nodes import credit_check_node
        state = _base_state(credit_score=500, annual_income=40000, existing_debt=2000)
        with self._patch_llm("Credit too low. [CREDIT_REJECTED: Credit score below 620]"):
            result = credit_check_node(state)
        self.assertFalse(result.get("credit_check_passed"))
        self.assertEqual(result.get("loan_decision"), "rejected")

    def test_property_valuation_pass(self):
        from nodes import property_valuation_node
        state = _base_state(
            property_value=400000,
            loan_amount=320000,
            down_payment=80000,
            property_address="123 Main St",
        )
        with self._patch_llm("Property checks out. [VALUATION_APPROVED]"):
            result = property_valuation_node(state)
        self.assertTrue(result.get("property_valuation_passed"))
        self.assertEqual(result.get("current_stage"), "underwriting")

    def test_property_valuation_fail(self):
        from nodes import property_valuation_node
        state = _base_state(
            property_value=300000,
            loan_amount=295000,
            down_payment=5000,
            property_address="456 Elm St",
        )
        with self._patch_llm("LTV too high. [VALUATION_REJECTED: LTV exceeds 95%]"):
            result = property_valuation_node(state)
        self.assertFalse(result.get("property_valuation_passed"))
        self.assertEqual(result.get("loan_decision"), "rejected")

    def test_underwriting_approval(self):
        from nodes import underwriting_node
        state = _base_state(
            applicant_name="Jane Doe",
            loan_amount=300000,
            credit_score=740,
            loan_term_years=30,
            annual_income=120000,
        )
        llm_content = (
            "Congratulations! "
            "[LOAN_APPROVED: amount=300000, rate=7.0, monthly=1995.91]"
        )
        with self._patch_llm(llm_content):
            result = underwriting_node(state)
        self.assertEqual(result.get("loan_decision"), "approved")
        self.assertAlmostEqual(result.get("approved_amount"), 300000.0)
        self.assertAlmostEqual(result.get("interest_rate"), 7.0)
        self.assertAlmostEqual(result.get("monthly_payment"), 1995.91)

    def test_underwriting_rejection(self):
        from nodes import underwriting_node
        state = _base_state(loan_amount=500000, annual_income=50000)
        with self._patch_llm("Cannot approve at this time."):
            result = underwriting_node(state)
        self.assertEqual(result.get("loan_decision"), "rejected")

    def test_loan_approval_node(self):
        from nodes import loan_approval_node
        state = _base_state(
            applicant_name="John Smith",
            approved_amount=350000,
            interest_rate=6.5,
            loan_term_years=30,
            monthly_payment=2212.24,
            property_address="789 Oak Ave",
        )
        with self._patch_llm("Congratulations, your loan is approved!"):
            result = loan_approval_node(state)
        self.assertEqual(result.get("current_stage"), "completed")

    def test_loan_rejection_node(self):
        from nodes import loan_rejection_node
        state = _base_state(
            applicant_name="Alice Brown",
            rejection_reason="Credit score too low",
        )
        with self._patch_llm("We regret to inform you..."):
            result = loan_rejection_node(state)
        self.assertEqual(result.get("current_stage"), "completed")


class TestApplicationSummary(unittest.TestCase):
    """_application_summary helper formats data correctly."""

    def test_empty_state_summary(self):
        from nodes import _application_summary
        state = _base_state()
        summary = _application_summary(state)
        self.assertIn("no data collected yet", summary)

    def test_partial_state_summary(self):
        from nodes import _application_summary
        state = _base_state(applicant_name="Bob", annual_income=75000, credit_score=720)
        summary = _application_summary(state)
        self.assertIn("Bob", summary)
        self.assertIn("75,000", summary)
        self.assertIn("720", summary)


if __name__ == "__main__":
    unittest.main()
