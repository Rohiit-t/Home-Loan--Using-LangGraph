"""Node functions for the Home Loan LangGraph agent."""

import re
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from prompts import (
    APPROVAL_PROMPT,
    CREDIT_CHECK_PROMPT,
    DOCUMENT_COLLECTION_PROMPT,
    ELIGIBILITY_CHECK_PROMPT,
    PROPERTY_VALUATION_PROMPT,
    REJECTION_PROMPT,
    UNDERWRITING_PROMPT,
)
from state import LoanState

# ---------------------------------------------------------------------------
# Shared LLM instance (temperature=0 for deterministic structured outputs)
# ---------------------------------------------------------------------------
_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _application_summary(state: LoanState) -> str:
    """Return a human-readable summary of the data collected so far."""
    lines = []
    fields = [
        ("Name", state.get("applicant_name")),
        ("Age", state.get("applicant_age")),
        ("Employment", state.get("employment_status")),
        ("Annual Income", f"${state['annual_income']:,.0f}" if state.get("annual_income") else None),
        ("Credit Score", state.get("credit_score")),
        ("Monthly Debt", f"${state['existing_debt']:,.0f}" if state.get("existing_debt") else None),
        ("Down Payment", f"${state['down_payment']:,.0f}" if state.get("down_payment") else None),
        ("Loan Amount", f"${state['loan_amount']:,.0f}" if state.get("loan_amount") else None),
        ("Loan Term", f"{state['loan_term_years']} years" if state.get("loan_term_years") else None),
        ("Property Address", state.get("property_address")),
        ("Property Value", f"${state['property_value']:,.0f}" if state.get("property_value") else None),
    ]
    for label, value in fields:
        if value is not None:
            lines.append(f"  {label}: {value}")
    return "\n".join(lines) if lines else "  (no data collected yet)"


def _format_messages(messages: list) -> str:
    """Format message list as a readable dialogue string."""
    parts = []
    for m in messages[-10:]:  # keep last 10 to avoid huge prompts
        if isinstance(m, HumanMessage):
            parts.append(f"Applicant: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Agent: {m.content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node: document_collection
# ---------------------------------------------------------------------------

def document_collection_node(state: LoanState) -> dict:
    """Conversational node that collects all required application data."""
    prompt = DOCUMENT_COLLECTION_PROMPT.format(
        application_summary=_application_summary(state),
        messages=_format_messages(state.get("messages", [])),
    )
    response = _llm.invoke(prompt)
    content: str = response.content

    updates: dict = {"messages": [AIMessage(content=content)]}

    # Parse any newly revealed data from the conversation
    updates.update(_extract_application_data(state, content))

    if "[DOCUMENTS_COMPLETE]" in content:
        updates["documents_collected"] = True
        updates["current_stage"] = "eligibility_check"

    return updates


def _extract_application_data(state: LoanState, text: str) -> dict:
    """
    Best-effort extraction of numeric/string fields from AI text.
    The LLM is expected to echo back collected data in the summary prompt,
    but we also try to parse values directly mentioned in the reply.
    """
    updates: dict = {}

    def _find_float(patterns):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return float(m.group(1).replace(",", ""))
        return None

    def _find_int(patterns):
        v = _find_float(patterns)
        return int(v) if v is not None else None

    def _find_str(patterns):
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    if not state.get("annual_income"):
        v = _find_float([r"annual income[^\$]*\$?([\d,]+)", r"income[^\$]*\$?([\d,]+)"])
        if v:
            updates["annual_income"] = v

    if not state.get("credit_score"):
        v = _find_int([r"credit score[^\d]*([\d]{3})", r"score[^\d]*([\d]{3})"])
        if v and 300 <= v <= 850:
            updates["credit_score"] = v

    if not state.get("loan_amount"):
        v = _find_float([r"loan amount[^\$]*\$?([\d,]+)", r"borrow[^\$]*\$?([\d,]+)"])
        if v:
            updates["loan_amount"] = v

    if not state.get("property_value"):
        v = _find_float([r"property value[^\$]*\$?([\d,]+)", r"home value[^\$]*\$?([\d,]+)"])
        if v:
            updates["property_value"] = v

    if not state.get("down_payment"):
        v = _find_float([r"down payment[^\$]*\$?([\d,]+)"])
        if v:
            updates["down_payment"] = v

    if not state.get("existing_debt"):
        v = _find_float([r"monthly debt[^\$]*\$?([\d,]+)", r"existing debt[^\$]*\$?([\d,]+)"])
        if v:
            updates["existing_debt"] = v

    if not state.get("loan_term_years"):
        v = _find_int([r"(\d+)[- ]year", r"loan term[^\d]*(\d+)"])
        if v in (15, 30):
            updates["loan_term_years"] = v

    return updates


# ---------------------------------------------------------------------------
# Node: eligibility_check
# ---------------------------------------------------------------------------

def eligibility_check_node(state: LoanState) -> dict:
    """Check basic eligibility criteria before deeper analysis."""
    prompt = ELIGIBILITY_CHECK_PROMPT.format(
        application_summary=_application_summary(state),
    )
    response = _llm.invoke(prompt)
    content: str = response.content

    updates: dict = {
        "messages": [AIMessage(content=content)],
        "current_stage": "eligibility_check",
    }

    if "[ELIGIBLE]" in content:
        updates["eligibility_passed"] = True
        updates["current_stage"] = "credit_check"
    else:
        reason_match = re.search(r"\[NOT_ELIGIBLE:\s*(.+?)\]", content)
        reason = reason_match.group(1) if reason_match else "Did not meet basic eligibility requirements."
        updates["eligibility_passed"] = False
        updates["loan_decision"] = "rejected"
        updates["rejection_reason"] = reason
        updates["current_stage"] = "rejection"

    return updates


# ---------------------------------------------------------------------------
# Node: credit_check
# ---------------------------------------------------------------------------

def credit_check_node(state: LoanState) -> dict:
    """Evaluate credit score and debt-to-income ratio."""
    annual_income = state.get("annual_income")
    if not annual_income:
        return {
            "messages": [AIMessage(content="Unable to perform credit check: annual income is missing.")],
            "credit_check_passed": False,
            "loan_decision": "rejected",
            "rejection_reason": "Annual income information is required for the credit check.",
            "current_stage": "rejection",
        }

    prompt = CREDIT_CHECK_PROMPT.format(
        application_summary=_application_summary(state),
        credit_score=state.get("credit_score", "unknown"),
        existing_debt=state.get("existing_debt", 0),
        annual_income=annual_income,
    )
    response = _llm.invoke(prompt)
    content: str = response.content

    updates: dict = {
        "messages": [AIMessage(content=content)],
        "current_stage": "credit_check",
    }

    if "[CREDIT_APPROVED]" in content:
        updates["credit_check_passed"] = True
        updates["current_stage"] = "property_valuation"
    else:
        reason_match = re.search(r"\[CREDIT_REJECTED:\s*(.+?)\]", content)
        reason = reason_match.group(1) if reason_match else "Credit check failed."
        updates["credit_check_passed"] = False
        updates["loan_decision"] = "rejected"
        updates["rejection_reason"] = reason
        updates["current_stage"] = "rejection"

    return updates


# ---------------------------------------------------------------------------
# Node: property_valuation
# ---------------------------------------------------------------------------

def property_valuation_node(state: LoanState) -> dict:
    """Validate LTV ratio and down payment adequacy."""
    prompt = PROPERTY_VALUATION_PROMPT.format(
        application_summary=_application_summary(state),
        property_address=state.get("property_address", "Not provided"),
        property_value=state.get("property_value", 0),
        loan_amount=state.get("loan_amount", 0),
        down_payment=state.get("down_payment", 0),
    )
    response = _llm.invoke(prompt)
    content: str = response.content

    updates: dict = {
        "messages": [AIMessage(content=content)],
        "current_stage": "property_valuation",
    }

    if "[VALUATION_APPROVED]" in content:
        updates["property_valuation_passed"] = True
        updates["current_stage"] = "underwriting"
    else:
        reason_match = re.search(r"\[VALUATION_REJECTED:\s*(.+?)\]", content)
        reason = reason_match.group(1) if reason_match else "Property valuation failed."
        updates["property_valuation_passed"] = False
        updates["loan_decision"] = "rejected"
        updates["rejection_reason"] = reason
        updates["current_stage"] = "rejection"

    return updates


# ---------------------------------------------------------------------------
# Node: underwriting
# ---------------------------------------------------------------------------

def underwriting_node(state: LoanState) -> dict:
    """Final underwriting decision and loan term calculation."""
    prompt = UNDERWRITING_PROMPT.format(
        application_summary=_application_summary(state),
    )
    response = _llm.invoke(prompt)
    content: str = response.content

    updates: dict = {
        "messages": [AIMessage(content=content)],
        "current_stage": "underwriting",
    }

    approval_match = re.search(
        r"\[LOAN_APPROVED:\s*amount=([\d.,]+),\s*rate=([\d.]+),\s*monthly=([\d.,]+)\]",
        content,
    )
    if approval_match:
        updates["loan_decision"] = "approved"
        updates["approved_amount"] = float(approval_match.group(1).replace(",", ""))
        updates["interest_rate"] = float(approval_match.group(2))
        updates["monthly_payment"] = float(approval_match.group(3).replace(",", ""))
        updates["current_stage"] = "approval"
    else:
        updates["loan_decision"] = "rejected"
        updates["rejection_reason"] = "Underwriting could not approve the loan at this time."
        updates["current_stage"] = "rejection"

    return updates


# ---------------------------------------------------------------------------
# Node: loan_approval
# ---------------------------------------------------------------------------

def loan_approval_node(state: LoanState) -> dict:
    """Generate the official approval letter."""
    prompt = APPROVAL_PROMPT.format(
        applicant_name=state.get("applicant_name", "Applicant"),
        approved_amount=state.get("approved_amount", 0),
        interest_rate=state.get("interest_rate", 0),
        loan_term_years=state.get("loan_term_years", 30),
        monthly_payment=state.get("monthly_payment", 0),
        property_address=state.get("property_address", "the property"),
    )
    response = _llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_stage": "completed",
    }


# ---------------------------------------------------------------------------
# Node: loan_rejection
# ---------------------------------------------------------------------------

def loan_rejection_node(state: LoanState) -> dict:
    """Generate the official rejection letter."""
    prompt = REJECTION_PROMPT.format(
        applicant_name=state.get("applicant_name", "Applicant"),
        rejection_reason=state.get("rejection_reason", "eligibility criteria not met"),
    )
    response = _llm.invoke(prompt)
    return {
        "messages": [AIMessage(content=response.content)],
        "current_stage": "completed",
    }
