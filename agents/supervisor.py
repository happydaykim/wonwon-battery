from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.base import build_agent_message, create_agent_blueprint
from config.settings import Settings, load_settings
from schemas.state import ReportState
from utils.logging import get_logger


SUPERVISOR_BLUEPRINT = create_agent_blueprint(
    name="supervisor_agent",
    prompt_name="supervisor.md",
)

logger = get_logger(__name__)

STEP_TO_PHASE_MAP = {
    "parallel_retrieval": "retrieve_market",
    "skeptic_lges": "skeptic_lges",
    "skeptic_catl": "skeptic_catl",
    "compare": "compare",
    "write": "write",
    "validate": "validate",
}

CANONICAL_PLAN_ORDER = (
    "parallel_retrieval",
    "skeptic_lges",
    "skeptic_catl",
    "compare",
    "write",
    "validate",
)
POST_RETRIEVAL_TAIL = ("compare", "write", "validate")

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUPERVISOR_BLUEPRINT.system_prompt),
        ("human", "{supervisor_context}"),
    ]
)


class SupervisorOutput(BaseModel):
    remaining_plan: list[str] = Field(description="다음 broad step queue")
    rationale: str = Field(description="선택 이유")


def supervisor_node(state: ReportState) -> dict:
    """Interpret the planner queue and prepare the next execution phase."""
    runtime = {**state["runtime"]}
    plan = state["plan"].copy()
    companies = {name: {**company_state} for name, company_state in state["companies"].items()}
    current_step = plan[0] if plan else None

    if current_step is None:
        runtime["current_phase"] = "done"
        note = "Planner queue is empty. Supervisor is marking the workflow as done."
    else:
        plan, decision_mode, rationale = _generate_supervisor_plan(state)
        current_step = plan[0] if plan else None
        if state["plan"] and state["plan"][0] == "parallel_retrieval" and _parallel_retrieval_completed(state):
            companies = _sync_skeptic_requirements(companies)

        if current_step is None:
            runtime["current_phase"] = "done"
            note = (
                f"Supervisor finalized the workflow via {decision_mode} decision. "
                f"{rationale}"
            )
        else:
            note = (
                f"Supervisor prepared the next execution step via {decision_mode} decision: "
                f"'{current_step}'. {rationale}"
            )

    if current_step is not None:
        runtime["current_phase"] = STEP_TO_PHASE_MAP[current_step]
        runtime["termination_reason"] = None

    message = build_agent_message(SUPERVISOR_BLUEPRINT.name, note)
    return {
        "plan": plan,
        "companies": companies,
        "messages": state["messages"] + [message],
        "runtime": runtime,
    }


def _create_supervisor_chain(settings: Settings) -> Any:
    supervisor_llm = init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=0,
    )
    return SUPERVISOR_PROMPT | supervisor_llm.with_structured_output(SupervisorOutput)


def _can_use_supervisor_llm(settings: Settings) -> bool:
    if settings.llm_provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return True


def _generate_supervisor_plan(state: ReportState) -> tuple[list[str], str, str]:
    settings = load_settings()
    if _can_use_supervisor_llm(settings):
        try:
            output = _create_supervisor_chain(settings).invoke(
                {"supervisor_context": _build_supervisor_context(state)}
            )
            sanitized_plan = _sanitize_supervisor_plan(output.remaining_plan, state)
            if sanitized_plan or not state["plan"]:
                return sanitized_plan, "llm", output.rationale.strip()
        except Exception as exc:  # pragma: no cover - depends on runtime credentials/network
            logger.warning("Supervisor LLM unavailable. Falling back to deterministic routing: %s", exc)

    fallback_plan = _fallback_supervisor_plan(state)
    return fallback_plan, "fallback", _build_fallback_rationale(state, fallback_plan)


def _parallel_retrieval_completed(state: ReportState) -> bool:
    """Check whether market/LGES/CATL retrieval bundle has finished."""
    current_step = state["plan"][0] if state["plan"] else None
    if current_step != "parallel_retrieval":
        return False

    market_ready = state["market"]["synthesized_summary"] is not None
    lges_ready = state["companies"]["LGES"]["synthesized_summary"] is not None
    catl_ready = state["companies"]["CATL"]["synthesized_summary"] is not None
    return market_ready and lges_ready and catl_ready


def _fallback_supervisor_plan(state: ReportState) -> list[str]:
    current_plan = state["plan"]
    current_step = current_plan[0] if current_plan else None
    if current_step == "parallel_retrieval" and _parallel_retrieval_completed(state):
        return _rewrite_post_retrieval_plan(state, current_plan[1:])
    return current_plan.copy()


def _rewrite_post_retrieval_plan(state: ReportState, remaining_plan: list[str]) -> list[str]:
    _ = remaining_plan
    return [*_required_skeptic_steps(state), *POST_RETRIEVAL_TAIL]


def _sync_skeptic_requirements(
    companies: dict[str, dict],
) -> dict[str, dict]:
    updated_companies = {name: {**company_state} for name, company_state in companies.items()}

    for company_state in updated_companies.values():
        requires_skeptic = not company_state["retrieval_sufficient"]
        company_state["skeptic_review_required"] = requires_skeptic
        if not requires_skeptic:
            company_state["skeptic_review_completed"] = False
            company_state["counter_evidence_ids"] = []

    return updated_companies


def _sanitize_supervisor_plan(raw_steps: list[str], state: ReportState) -> list[str]:
    sanitized = [step for step in CANONICAL_PLAN_ORDER if step in {item.strip() for item in raw_steps}]
    current_plan = state["plan"]
    if not current_plan:
        return []

    current_step = current_plan[0]
    if current_step == "parallel_retrieval" and _parallel_retrieval_completed(state):
        return _rewrite_post_retrieval_plan(state, sanitized)

    current_index = CANONICAL_PLAN_ORDER.index(current_step)
    suffix = [
        step
        for step in sanitized
        if CANONICAL_PLAN_ORDER.index(step) >= current_index
    ]
    if suffix and suffix[0] == current_step:
        return suffix
    return current_plan.copy()


def _build_supervisor_context(state: ReportState) -> str:
    current_plan = state["plan"]
    current_step = current_plan[0] if current_plan else "done"
    validation_issues = state.get("validation_issues", [])
    retryable_issue_count = sum(1 for issue in validation_issues if issue["retryable"])
    non_retryable_issue_count = len(validation_issues) - retryable_issue_count
    return "\n".join(
        [
            f"current_step={current_step}",
            f"current_plan={current_plan}",
            f"rewrite_allowed={current_step == 'parallel_retrieval' and _parallel_retrieval_completed(state)}",
            f"termination_reason={state['runtime'].get('termination_reason')}",
            f"revision_count={state['runtime']['revision_count']}/{state['runtime']['max_revisions']}",
            f"final_report_present={bool(state.get('final_report'))}",
            "[market]",
            _format_research_state(
                sufficient=state["market"]["retrieval_sufficient"],
                gaps=state["market"]["retrieval_gaps"],
                used_web_search=state["market"]["used_web_search"],
            ),
            "[LGES]",
            _format_research_state(
                sufficient=state["companies"]["LGES"]["retrieval_sufficient"],
                gaps=state["companies"]["LGES"]["retrieval_gaps"],
                used_web_search=state["companies"]["LGES"]["used_web_search"],
            ),
            "[CATL]",
            _format_research_state(
                sufficient=state["companies"]["CATL"]["retrieval_sufficient"],
                gaps=state["companies"]["CATL"]["retrieval_gaps"],
                used_web_search=state["companies"]["CATL"]["used_web_search"],
            ),
            "[validation]",
            f"retryable_issue_count={retryable_issue_count}",
            f"non_retryable_issue_count={non_retryable_issue_count}",
            "[allowed_steps]",
            *CANONICAL_PLAN_ORDER,
        ]
    )


def _build_fallback_rationale(state: ReportState, plan: list[str]) -> str:
    current_step = state["plan"][0] if state["plan"] else None
    if current_step == "parallel_retrieval" and _parallel_retrieval_completed(state):
        return _build_post_retrieval_note(state, plan)
    if current_step is None:
        return "No remaining broad steps."
    return f"Preserved the in-flight queue semantics for step '{current_step}'."


def _required_skeptic_steps(state: ReportState) -> list[str]:
    skeptic_steps: list[str] = []
    for company, skeptic_step in (("LGES", "skeptic_lges"), ("CATL", "skeptic_catl")):
        if not state["companies"][company]["retrieval_sufficient"]:
            skeptic_steps.append(skeptic_step)
    return skeptic_steps


def _format_research_state(
    *,
    sufficient: bool,
    gaps: list[str],
    used_web_search: bool,
) -> str:
    gap_text = "; ".join(gaps) if gaps else "none"
    return f"sufficient={sufficient}, used_web_search={used_web_search}, gaps={gap_text}"


def _build_post_retrieval_note(state: ReportState, plan: list[str]) -> str:
    company_notes = []
    for company in ("LGES", "CATL"):
        company_state = state["companies"][company]
        status = "sufficient" if company_state["retrieval_sufficient"] else "needs_skeptic_review"
        company_notes.append(f"{company}={status}")

    market_note = (
        "MARKET=sufficient"
        if state["market"]["retrieval_sufficient"]
        else "MARKET=insufficient"
    )
    return (
        "Parallel retrieval bundle completed. "
        f"Post-retrieval branch prepared with {market_note}, "
        f"{', '.join(company_notes)}. "
        f"Next queue: {', '.join(plan) if plan else 'done'}."
    )
