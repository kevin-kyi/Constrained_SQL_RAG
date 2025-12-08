from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MidEGDConfig:
    num_beams: int = 4
    max_new_tokens: int = 160
    egd_interval: int = 16          # run EGD every N generated tokens
    min_tokens_for_egd: int = 32    # don't run EGD before this many new tokens
    egd_top_k: int = 2              # only check top-K beams at each EGD step


@dataclass
class BeamState:
    input_ids: torch.LongTensor  # 1D tensor on model device
    logprob: float
    finished: bool = False
    killed: bool = False
    num_egd_checks: int = 0
    num_egd_failures: int = 0


@dataclass
class MidEGDBeamResult:
    text: str
    sql: str
    logprob: float
    killed: bool
    num_egd_checks: int
    num_egd_failures: int


def _get_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def find_executable_sql_prefix(text: str) -> Optional[str]:
    """
    Given the *decoded* text of a beam, try to extract a *complete* SQL query
    from the first SELECT up to the last semicolon ';'.

    Requirements:
      - There is at least one 'select ' (case-insensitive)
      - There is at least one ';' after that SELECT
      - Parentheses inside that span are balanced

    Returns:
      - SQL string (trimmed) if we think it's executable
      - None otherwise
    """
    lower = text.lower()
    first_select = lower.find("select ")
    last_semi = text.rfind(";")

    if first_select == -1 or last_semi == -1 or last_semi <= first_select:
        return None

    candidate = text[first_select : last_semi + 1]

    # Very cheap paren balancing check
    if candidate.count("(") != candidate.count(")"):
        return None

    return candidate.strip()


def generate_sql_mid_egd(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    db_id: str,
    execute_sql_fn: Callable[[str, str], Dict[str, Any]],
    extract_sql_fn: Callable[[str], str],
    config: Optional[MidEGDConfig] = None,
) -> Tuple[str, List[MidEGDBeamResult]]:
    """
    Manual beam search with mid-decoding execution-guided pruning.

    Args
    ----
    model:
        AutoModelForCausalLM instance (SQLCoder).
    tokenizer:
        Matching tokenizer.
    prompt:
        Full text prompt given to the model.
    db_id:
        Spider DB id, used by execute_sql_fn.
    execute_sql_fn:
        Function (db_id, sql) -> {'ok': bool, 'rows': [...]}.
    extract_sql_fn:
        Function text -> final SQL string (for full generations).
    config:
        MidEGDConfig with beam + EGD parameters.

    Returns
    -------
    best_sql: str
        SQL string for the best beam at the end of decoding.
    beam_debug: List[MidEGDBeamResult]
        Debug info for all final beams.
    """
    if config is None:
        config = MidEGDConfig()

    device = _get_device(model)
    model.eval()

    with torch.no_grad():
        # Encode prompt once
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)[0]  # (L,)

        eos_id = tokenizer.eos_token_id

        # Initialize a single root beam
        beams: List[BeamState] = [
            BeamState(input_ids=input_ids.clone(), logprob=0.0)
        ]

        # Main decoding loop over new tokens
        for step in range(config.max_new_tokens):
            candidate_beams: List[BeamState] = []

            for beam in beams:
                # If beam is finished or killed, just carry it forward unchanged
                if beam.finished or beam.killed:
                    candidate_beams.append(beam)
                    continue

                # Run model on the full current sequence for this beam
                outputs = model(
                    input_ids=beam.input_ids.unsqueeze(0),
                    use_cache=False,
                )
                logits = outputs.logits[0, -1, :]  # last token logits
                logprobs = F.log_softmax(logits, dim=-1)

                # Expand with top-K tokens for this beam
                topk_logprobs, topk_ids = torch.topk(
                    logprobs, k=config.num_beams
                )

                for j in range(topk_ids.size(0)):
                    token_id = topk_ids[j].unsqueeze(0)  # (1,)
                    new_ids = torch.cat([beam.input_ids, token_id], dim=0)
                    new_lp = beam.logprob + topk_logprobs[j].item()

                    new_finished = bool(
                        eos_id is not None
                        and int(token_id.item()) == int(eos_id)
                    )

                    candidate_beams.append(
                        BeamState(
                            input_ids=new_ids,
                            logprob=new_lp,
                            finished=new_finished,
                            killed=beam.killed,
                            num_egd_checks=beam.num_egd_checks,
                            num_egd_failures=beam.num_egd_failures,
                        )
                    )

            # Prune back to num_beams by logprob (ignoring killed/finished for now)
            candidate_beams.sort(key=lambda b: b.logprob, reverse=True)
            beams = candidate_beams[: config.num_beams]

            # Mid-decoding EGD: periodically run execution checks on top beams
            new_tokens_generated = step + 1
            if (
                new_tokens_generated >= config.min_tokens_for_egd
                and new_tokens_generated % config.egd_interval == 0
            ):
                # Only consider alive beams
                alive_beams = [b for b in beams if not b.killed]

                # If nothing is alive, no point in running EGD
                if alive_beams:
                    # Take top-K alive beams by logprob
                    alive_beams.sort(key=lambda b: b.logprob, reverse=True)
                    egd_targets = alive_beams[: config.egd_top_k]

                    for b in egd_targets:
                        # Decode prefix
                        prefix_text = tokenizer.decode(
                            b.input_ids, skip_special_tokens=True
                        )
                        # Find an executable prefix if any
                        partial_sql = find_executable_sql_prefix(prefix_text)
                        if partial_sql is None:
                            continue

                        b.num_egd_checks += 1
                        exec_res = execute_sql_fn(db_id, partial_sql)
                        if not exec_res.get("ok", False):
                            # Mark beam as dead; it will be pruned out soon
                            b.num_egd_failures += 1
                            b.killed = True
                            b.logprob = -1e9

            # Early stop if all beams are finished or killed
            if all(b.finished or b.killed for b in beams):
                break

        # End of decoding: choose best beam among non-killed
        alive_beams = [b for b in beams if not b.killed]
        if alive_beams:
            best = max(alive_beams, key=lambda b: b.logprob)
        else:
            # Fallback: if all killed, pick the least-bad beam overall
            best = max(beams, key=lambda b: b.logprob)

        # Decode final text + extract final SQL
        best_text = tokenizer.decode(best.input_ids, skip_special_tokens=True)
        best_sql = extract_sql_fn(best_text)

        # Build debug info
        beam_debug: List[MidEGDBeamResult] = []
        for b in beams:
            txt = tokenizer.decode(b.input_ids, skip_special_tokens=True)
            sql = extract_sql_fn(txt)
            beam_debug.append(
                MidEGDBeamResult(
                    text=txt,
                    sql=sql,
                    logprob=b.logprob,
                    killed=b.killed,
                    num_egd_checks=b.num_egd_checks,
                    num_egd_failures=b.num_egd_failures,
                )
            )

        return best_sql, beam_debug
