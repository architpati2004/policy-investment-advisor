"""Question-answering endpoints.

Both routes return **200 for a refusal**, with the refusal text in the body.
That is the rule CLAUDE.md section 11 draws from measurement: asked whether bank
deposit rules affect an FMCG holding, qwen3:4b replies "the sources govern
commercial banks and do not cover NBFCs" *as* a refusal, and the same content
from qwen3:1.7b arrives as an answer. The two models disagree about the label
and agree about the finding, so a client that hides the body when ``covered`` is
false throws away the useful part. An error status here would be a lie as well
as a loss: nothing failed.

Routes are ``def`` rather than ``async def`` deliberately. Generation is a
blocking call of thirty seconds or more, and FastAPI runs sync handlers in a
worker thread — declaring them async would block the event loop and stall every
other request for the duration.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api import schemas
from backend.api.deps import get_db, get_policy_rag, get_portfolio_rag
from backend.db import portfolio as portfolio_service
from backend.rag.policy_rag import PolicyRAG
from backend.rag.portfolio_rag import PortfolioRAG

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/policy", response_model=schemas.AnswerOut)
def ask_policy(payload: schemas.QuestionIn, rag: PolicyRAG = Depends(get_policy_rag)) -> schemas.AnswerOut:
    """Answer a policy question from the indexed regulation, or decline to.

    Slow: thirty seconds or more on local inference, and the first call of a
    session also loads the model into RAM.
    """
    return schemas.AnswerOut(**rag.ask(payload.question).to_dict())


@router.post("/portfolio", response_model=schemas.AssessmentOut)
def assess_portfolio(
    payload: schemas.PortfolioQuestionIn,
    rag: PortfolioRAG = Depends(get_portfolio_rag),
    session: Session = Depends(get_db),
) -> schemas.AssessmentOut:
    """Assess a question against the portfolio's holdings."""
    assessment = rag.assess(
        payload.question,
        session,
        portfolio_name=payload.portfolio or portfolio_service.DEFAULT_PORTFOLIO,
    )
    return schemas.AssessmentOut(**assessment.to_dict())
