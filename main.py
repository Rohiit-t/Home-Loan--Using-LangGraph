"""
Home Loan Agent — Main Entry Point
====================================
Run this script to start an interactive home loan application session.

Usage:
    python main.py

Environment variables (set in a .env file or export before running):
    OPENAI_API_KEY — your OpenAI API key
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# Validate API key early
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. "
        "Please set it in your environment or in a .env file."
    )

from graph import home_loan_graph  # noqa: E402 — import after env check
from state import LoanState        # noqa: E402


BANNER = """
╔══════════════════════════════════════════════════════════╗
║         Welcome to the AI Home Loan Agent  🏠            ║
║  Powered by LangGraph — your smart mortgage assistant    ║
╚══════════════════════════════════════════════════════════╝
Type your message and press Enter. Type 'quit' to exit.
"""


def _initial_state() -> LoanState:
    """Return a fresh LoanState with default values."""
    return LoanState(
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


def _print_last_ai_message(state: LoanState) -> None:
    """Print the most recent AI message from the state."""
    for msg in reversed(state.get("messages", [])):
        from langchain_core.messages import AIMessage
        if isinstance(msg, AIMessage):
            print(f"\n🏦 Agent: {msg.content}\n")
            break


def run() -> None:
    """Run the interactive home loan agent session."""
    print(BANNER)

    state = _initial_state()
    # Kick off the first agent turn with an empty greeting trigger
    state = home_loan_graph.invoke(
        {**state, "messages": [HumanMessage(content="Hello, I'd like to apply for a home loan.")]},
        {"recursion_limit": 100},
    )
    _print_last_ai_message(state)

    while True:
        # Check if workflow has reached a terminal stage
        if state.get("current_stage") == "completed":
            print("✅ Your application process is complete. Thank you!")
            break

        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye! Feel free to come back when you're ready to apply.")
            break

        # Append the human message and invoke the graph
        state = home_loan_graph.invoke(
            {**state, "messages": state.get("messages", []) + [HumanMessage(content=user_input)]},
            {"recursion_limit": 100},
        )
        _print_last_ai_message(state)

        # After terminal nodes, the stage will be "completed"
        if state.get("current_stage") == "completed":
            print("✅ Your application process is complete. Thank you!")
            break


if __name__ == "__main__":
    run()
