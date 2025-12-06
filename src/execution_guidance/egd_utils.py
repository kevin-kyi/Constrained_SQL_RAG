from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

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
    """
    sql: str
    text: str
    logprob: float
    prob: float
    exec_result: Dict[str, Any]
    score: float


def execution_score(exec_result: Dict[str, Any]) -> float:
    """
    Simple execution-based scoring heuristic.

    - 0.0  if SQL failed to execute (syntax/runtime error).
    - 0.5  if SQL executed but returned an empty result.
    - 1.0  if SQL executed and returned at least one row.

    This is intentionally simple for the "easy" EGD version.
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

    # Numerically stable softmax.
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


def apply_egd_reranking(
    db_id: str,
    raw_candidates: List[BeamCandidateLike],
    extract_sql_fn: Callable[[str], str],
    execute_sql_fn: Callable[[str, str], Dict[str, Any]],
) -> Tuple[EGDCandidate, List[EGDCandidate]]:
    """
    Apply simple execution-guided reranking over a list of raw beam candidates.

    Parameters
    ----------
    db_id:
        Spider database ID for this example.
    raw_candidates:
        List of beam-level candidates (BeamCandidate). Each must expose
        `.text` (decoded text) and `.logprob` (scalar float).
    extract_sql_fn:
        Function that extracts a clean SQL string from the decoded text.
        (We re-use eval_sqlcoder.extract_sql for this.)
    execute_sql_fn:
        Function that executes (db_id, sql) and returns a result dict:
            - ok: bool
            - rows: list[tuple] (optional if ok=False)
            - error: str (optional)

    Returns
    -------
    best_candidate:
        The highest-scoring candidate according to execution_score (tie-broken
        by probability / logprob).
    all_candidates:
        List of EGDCandidate with execution results and scores for logging.
    """
    if not raw_candidates:
        raise ValueError("apply_egd_reranking() called with no candidates")

    # 1) Normalize logprobs across the candidate set for nicer logging.
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
            )
        )

    # 2) Choose best according to (execution_score, probability, logprob)
    best = max(
        egd_candidates,
        key=lambda c: (c.score, c.prob, c.logprob),
    )

    return best, egd_candidates
