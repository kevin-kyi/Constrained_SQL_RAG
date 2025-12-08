from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import math


@dataclass
class EGDCandidate:
    """
    Container for a single EGD candidate after execution-guided scoring.

    Attributes
    ----------
    sql:
        The cleaned SQL string for this candidate.
    text:
        The raw decoded text (may include extra context depending on extraction).
    logprob:
        Log-probability of the candidate (up to a constant; higher is better).
    prob:
        Normalized probability across the candidate set (softmax over logprobs).
    exec_result:
        Result dict from execute_sql(db_id, sql).
    score:
        Execution-guided score used for ranking.
    schema_ok:
        Whether this candidate passed schema checks (used in "medium" EGD).
    """
    sql: str
    text: str
    logprob: float
    prob: float
    exec_result: Dict[str, Any]
    score: float
    schema_ok: bool = True


def execution_score(exec_result: Dict[str, Any]) -> float:
    """
    Simple execution-based scoring heuristic.

    - 0.0  if SQL failed to execute (syntax/runtime error).
    - 0.5  if SQL executed but returned an empty result.
    - 1.0  if SQL executed and returned at least one row.
    """
    if not exec_result.get("ok", False):
        return 0.0
    rows = exec_result.get("rows", [])
    return 1.0 if rows else 0.5


def normalize_logprobs(logprobs: List[float]) -> List[float]:
    """
    Convert a list of (unnormalized) log-probabilities into a proper probability
    distribution with softmax. This is just for interpretability / logging.
    """
    if not logprobs:
        return []

    m = max(logprobs)
    exps = [math.exp(lp - m) for lp in logprobs]
    s = sum(exps)
    if s == 0.0:
        n = len(logprobs)
        return [1.0 / n] * n
    return [v / s for v in exps]


class BeamCandidateLike:
    """
    Minimal protocol-style class for type hints.

    In the actual project code, `raw_candidates` will be a list of
    `BeamCandidate` objects defined in `generators.py`. We only rely
    on two attributes: `.text` and `.logprob`.
    """
    text: str
    logprob: float


# -------------------------------------------------------------------
# Simple EGD: execution-only reranking (what you already have running)
# -------------------------------------------------------------------
def apply_egd_reranking(
    db_id: str,
    raw_candidates: List[BeamCandidateLike],
    extract_sql_fn: Callable[[str], str],
    execute_sql_fn: Callable[[str, str], Dict[str, Any]],
):
    """
    Execution-only EGD: rerank beam candidates based on execution_result.

    This is the "cheap" version:
    - no schema checks
    - scores candidates only by execution success / emptiness
    """
    if not raw_candidates:
        raise ValueError("apply_egd_reranking() called with no candidates")

    logprobs = [float(c.logprob) for c in raw_candidates]
    probs = normalize_logprobs(logprobs)

    egd_candidates: List[EGDCandidate] = []
    for cand, lp, p in zip(raw_candidates, logprobs, probs):
        text = cand.text
        sql = extract_sql_fn(text)
        exec_result = execute_sql_fn(db_id, sql)
        score = execution_score(exec_result)
        egd_candidates.append(
            EGDCandidate(
                sql=sql,
                text=text,
                logprob=lp,
                prob=p,
                exec_result=exec_result,
                score=score,
                schema_ok=True,
            )
        )

    best = max(egd_candidates, key=lambda c: (c.score, c.prob, c.logprob))
    return best, egd_candidates


# -------------------------------------------------------------------
# Medium EGD-lite: schema-aware + execution-based reranking
# -------------------------------------------------------------------
def apply_egd_reranking_with_schema(
    db_id: str,
    raw_candidates: List[BeamCandidateLike],
    extract_sql_fn: Callable[[str], str],
    execute_sql_fn: Callable[[str, str], Dict[str, Any]],
    schema_check_fn: Callable[[str, str], bool],
):
    """
    Schema-aware EGD reranking.

    For each candidate:
      - Extract SQL string
      - Run a cheap schema check: schema_check_fn(db_id, sql) -> bool
      - Execute SQL and compute execution_score
      - Combine them into a final score:
            if not schema_ok: score = 0.0
            else:             score = execution_score(exec_result)

    Returns:
      best_candidate, all_candidates
    """
    if not raw_candidates:
        raise ValueError("apply_egd_reranking_with_schema() called with no candidates")

    logprobs = [float(c.logprob) for c in raw_candidates]
    probs = normalize_logprobs(logprobs)

    egd_candidates: List[EGDCandidate] = []
    for cand, lp, p in zip(raw_candidates, logprobs, probs):
        text = cand.text
        sql = extract_sql_fn(text)
        schema_ok = schema_check_fn(db_id, sql)
        exec_result = execute_sql_fn(db_id, sql)

        if not schema_ok:
            score = 0.0
        else:
            score = execution_score(exec_result)

        egd_candidates.append(
            EGDCandidate(
                sql=sql,
                text=text,
                logprob=lp,
                prob=p,
                exec_result=exec_result,
                score=score,
                schema_ok=schema_ok,
            )
        )

    best = max(egd_candidates, key=lambda c: (c.score, c.prob, c.logprob))
    return best, egd_candidates
