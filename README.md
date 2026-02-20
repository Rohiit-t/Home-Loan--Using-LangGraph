# Home Loan Agent — Powered by LangGraph 🏠

An AI-driven home loan application assistant that orchestrates the full mortgage workflow using [LangGraph](https://github.com/langchain-ai/langgraph) and OpenAI's GPT-4o-mini.

---

## Workflow Overview

```
START
  │
  ▼
document_collection  ──(loop until complete)──►  document_collection
  │ (all info gathered)
  ▼
eligibility_check  ──(fail)──► loan_rejection ──► END
  │ (pass)
  ▼
credit_check  ──(fail)──► loan_rejection ──► END
  │ (pass)
  ▼
property_valuation  ──(fail)──► loan_rejection ──► END
  │ (pass)
  ▼
underwriting  ──(fail)──► loan_rejection ──► END
  │ (approved)
  ▼
loan_approval ──► END
```

### Stages

| Stage | Description |
|---|---|
| **Document Collection** | Conversational node that gathers all required applicant data |
| **Eligibility Check** | Age ≥ 18, employed/self-employed, income ≥ $30k, LTV ≤ 90% |
| **Credit Check** | Credit score ≥ 620, Debt-to-Income ratio ≤ 43% |
| **Property Valuation** | LTV ≤ 95%, down payment ≥ 5% of property value |
| **Underwriting** | Final decision + calculates interest rate and monthly payment |
| **Loan Approval / Rejection** | Generates a professional letter with full details or improvement suggestions |

---

## Project Structure

```
├── state.py          # LoanState TypedDict (graph state schema)
├── prompts.py        # Prompt templates for each workflow stage
├── nodes.py          # Node functions (LLM calls + state updates)
├── graph.py          # LangGraph workflow construction & compilation
├── main.py           # Interactive CLI entry point
├── tests.py          # Unit tests (no API key required)
└── requirements.txt  # Python dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
# or create a .env file:
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Run the agent

```bash
python main.py
```

---

## Running Tests

Tests use mocked LLM responses and require no API key:

```bash
pip install pytest
python -m pytest tests.py -v
```

---

## Key Design Decisions

- **LangGraph `StateGraph`** — each mortgage stage is a node; conditional edges route based on pass/fail flags in `LoanState`.
- **Append-only messages** — uses LangGraph's built-in `add_messages` reducer so conversation history accumulates safely across invocations.
- **Deterministic LLM outputs** — all nodes use `temperature=0` and structured sentinel tokens (e.g. `[ELIGIBLE]`, `[CREDIT_APPROVED]`) to make routing reliable.
- **Stateless nodes** — every node reads from and writes to `LoanState` only, making the workflow easy to test and extend.
