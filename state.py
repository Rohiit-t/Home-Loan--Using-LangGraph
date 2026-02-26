"""State definition for the Home Loan LangGraph agent."""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class LoanState(TypedDict):
    """Tracks the complete state of a home loan application."""

    # Conversation history (append-only via add_messages reducer)
    messages: Annotated[list, add_messages]

    # Personal information
    applicant_name: Optional[str]
    applicant_age: Optional[int]
    employment_status: Optional[str]  # "employed", "self-employed", "unemployed"
    annual_income: Optional[float]

    # Financial information
    credit_score: Optional[int]
    existing_debt: Optional[float]       # monthly existing debt obligations
    down_payment: Optional[float]

    # Loan details
    loan_amount: Optional[float]
    loan_term_years: Optional[int]       # e.g. 15 or 30
    property_value: Optional[float]
    property_address: Optional[str]

    # Workflow tracking
    current_stage: str                   # current workflow stage
    documents_collected: bool
    eligibility_passed: bool
    credit_check_passed: bool
    property_valuation_passed: bool

    # Outcome
    loan_decision: Optional[str]         # "approved", "rejected", "pending"
    rejection_reason: Optional[str]
    approved_amount: Optional[float]
    interest_rate: Optional[float]
    monthly_payment: Optional[float]
