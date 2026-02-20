"""Build the Home Loan LangGraph workflow."""

from langgraph.graph import END, START, StateGraph

from nodes import (
    credit_check_node,
    document_collection_node,
    eligibility_check_node,
    loan_approval_node,
    loan_rejection_node,
    property_valuation_node,
    underwriting_node,
)
from state import LoanState


# ---------------------------------------------------------------------------
# Routing helpers (conditional edges)
# ---------------------------------------------------------------------------

def route_after_documents(state: LoanState) -> str:
    """After document collection: proceed to eligibility or keep collecting."""
    if state.get("documents_collected"):
        return "eligibility_check"
    return "document_collection"


def route_after_eligibility(state: LoanState) -> str:
    """After eligibility check: proceed or reject."""
    if state.get("eligibility_passed"):
        return "credit_check"
    return "loan_rejection"


def route_after_credit(state: LoanState) -> str:
    """After credit check: proceed or reject."""
    if state.get("credit_check_passed"):
        return "property_valuation"
    return "loan_rejection"


def route_after_valuation(state: LoanState) -> str:
    """After property valuation: proceed or reject."""
    if state.get("property_valuation_passed"):
        return "underwriting"
    return "loan_rejection"


def route_after_underwriting(state: LoanState) -> str:
    """After underwriting: approve or reject."""
    if state.get("loan_decision") == "approved":
        return "loan_approval"
    return "loan_rejection"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the home loan LangGraph workflow."""
    graph = StateGraph(LoanState)

    # Register nodes
    graph.add_node("document_collection", document_collection_node)
    graph.add_node("eligibility_check", eligibility_check_node)
    graph.add_node("credit_check", credit_check_node)
    graph.add_node("property_valuation", property_valuation_node)
    graph.add_node("underwriting", underwriting_node)
    graph.add_node("loan_approval", loan_approval_node)
    graph.add_node("loan_rejection", loan_rejection_node)

    # Entry point
    graph.add_edge(START, "document_collection")

    # Conditional edges
    graph.add_conditional_edges(
        "document_collection",
        route_after_documents,
        {
            "document_collection": "document_collection",
            "eligibility_check": "eligibility_check",
        },
    )
    graph.add_conditional_edges(
        "eligibility_check",
        route_after_eligibility,
        {
            "credit_check": "credit_check",
            "loan_rejection": "loan_rejection",
        },
    )
    graph.add_conditional_edges(
        "credit_check",
        route_after_credit,
        {
            "property_valuation": "property_valuation",
            "loan_rejection": "loan_rejection",
        },
    )
    graph.add_conditional_edges(
        "property_valuation",
        route_after_valuation,
        {
            "underwriting": "underwriting",
            "loan_rejection": "loan_rejection",
        },
    )
    graph.add_conditional_edges(
        "underwriting",
        route_after_underwriting,
        {
            "loan_approval": "loan_approval",
            "loan_rejection": "loan_rejection",
        },
    )

    # Terminal nodes
    graph.add_edge("loan_approval", END)
    graph.add_edge("loan_rejection", END)

    return graph.compile()


# Compile once for import reuse
home_loan_graph = build_graph()
