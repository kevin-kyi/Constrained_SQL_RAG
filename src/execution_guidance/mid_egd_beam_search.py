# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, Optional, Tuple

# import torch
# import torch.nn.functional as F


# @dataclass
# class MidEGDConfig:
#     num_beams: int = 4
#     max_new_tokens: int = 160
#     egd_interval: int = 16         
#     min_tokens_for_egd: int = 32    
#     egd_top_k: int = 2              


# @dataclass
# class BeamState:
#     input_ids: torch.LongTensor  
#     logprob: float
#     finished: bool = False
#     killed: bool = False
#     num_egd_checks: int = 0
#     num_egd_failures: int = 0


# @dataclass
# class MidEGDBeamResult:
#     text: str
#     sql: str
#     logprob: float
#     killed: bool
#     num_egd_checks: int
#     num_egd_failures: int


# def _get_device(model: torch.nn.Module) -> torch.device:
#     try:
#         return next(model.parameters()).device
#     except StopIteration:
#         return torch.device("cpu")


# def find_executable_sql_prefix(text: str) -> Optional[str]:
#     """
#     Given the *decoded* text of a beam, try to extract a *complete* SQL query
#     from the first SELECT up to the last semicolon ';'.

#     Requirements:
#       - There is at least one 'select ' (case-insensitive)
#       - There is at least one ';' after that SELECT
#       - Parentheses inside that span are balanced

#     Returns:
#       - SQL string (trimmed) if we think it's executable
#       - None otherwise
#     """
#     lower = text.lower()
#     first_select = lower.find("select ")
#     last_semi = text.rfind(";")

#     if first_select == -1 or last_semi == -1 or last_semi <= first_select:
#         return None

#     candidate = text[first_select : last_semi + 1]

#     # Very cheap paren balancing check
#     if candidate.count("(") != candidate.count(")"):
#         return None

#     return candidate.strip()


# def generate_sql_mid_egd(
#     model: torch.nn.Module,
#     tokenizer,
#     prompt: str,
#     db_id: str,
#     execute_sql_fn: Callable[[str, str], Dict[str, Any]],
#     extract_sql_fn: Callable[[str], str],
#     config: Optional[MidEGDConfig] = None,
# ) -> Tuple[str, List[MidEGDBeamResult]]:
#     """
#     Manual beam search with mid-decoding execution-guided pruning.

#     Args
#     ----
#     model:
#         AutoModelForCausalLM instance (SQLCoder).
#     tokenizer:
#         Matching tokenizer.
#     prompt:
#         Full text prompt given to the model.
#     db_id:
#         Spider DB id, used by execute_sql_fn.
#     execute_sql_fn:
#         Function (db_id, sql) -> {'ok': bool, 'rows': [...]}.
#     extract_sql_fn:
#         Function text -> final SQL string (for full generations).
#     config:
#         MidEGDConfig with beam + EGD parameters.

#     Returns
#     -------
#     best_sql: str
#         SQL string for the best beam at the end of decoding.
#     beam_debug: List[MidEGDBeamResult]
#         Debug info for all final beams.
#     """
#     if config is None:
#         config = MidEGDConfig()

#     device = _get_device(model)
#     model.eval()

#     with torch.no_grad():
#         # Encode prompt once
#         enc = tokenizer(prompt, return_tensors="pt")
#         input_ids = enc["input_ids"].to(device)[0]  # (L,)

#         eos_id = tokenizer.eos_token_id

#         # Initialize a single root beam
#         beams: List[BeamState] = [
#             BeamState(input_ids=input_ids.clone(), logprob=0.0)
#         ]

#         # Main decoding loop over new tokens
#         for step in range(config.max_new_tokens):
#             candidate_beams: List[BeamState] = []

#             for beam in beams:
#                 # If beam is finished or killed, just carry it forward unchanged
#                 if beam.finished or beam.killed:
#                     candidate_beams.append(beam)
#                     continue

#                 # Run model on the full current sequence for this beam
#                 outputs = model(
#                     input_ids=beam.input_ids.unsqueeze(0),
#                     use_cache=False,
#                 )
#                 logits = outputs.logits[0, -1, :]  # last token logits
#                 logprobs = F.log_softmax(logits, dim=-1)

#                 # Expand with top-K tokens for this beam
#                 topk_logprobs, topk_ids = torch.topk(
#                     logprobs, k=config.num_beams
#                 )

#                 for j in range(topk_ids.size(0)):
#                     token_id = topk_ids[j].unsqueeze(0)  # (1,)
#                     new_ids = torch.cat([beam.input_ids, token_id], dim=0)
#                     new_lp = beam.logprob + topk_logprobs[j].item()

#                     new_finished = bool(
#                         eos_id is not None
#                         and int(token_id.item()) == int(eos_id)
#                     )

#                     candidate_beams.append(
#                         BeamState(
#                             input_ids=new_ids,
#                             logprob=new_lp,
#                             finished=new_finished,
#                             killed=beam.killed,
#                             num_egd_checks=beam.num_egd_checks,
#                             num_egd_failures=beam.num_egd_failures,
#                         )
#                     )

#             # Prune back to num_beams by logprob (ignoring killed/finished for now)
#             candidate_beams.sort(key=lambda b: b.logprob, reverse=True)
#             beams = candidate_beams[: config.num_beams]

#             # Mid-decoding EGD: periodically run execution checks on top beams
#             new_tokens_generated = step + 1
#             if (
#                 new_tokens_generated >= config.min_tokens_for_egd
#                 and new_tokens_generated % config.egd_interval == 0
#             ):
#                 # Only consider alive beams
#                 alive_beams = [b for b in beams if not b.killed]

#                 # If nothing is alive, no point in running EGD
#                 if alive_beams:
#                     # Take top-K alive beams by logprob
#                     alive_beams.sort(key=lambda b: b.logprob, reverse=True)
#                     egd_targets = alive_beams[: config.egd_top_k]

#                     for b in egd_targets:
#                         # Decode prefix
#                         prefix_text = tokenizer.decode(
#                             b.input_ids, skip_special_tokens=True
#                         )
#                         # Find an executable prefix if any
#                         partial_sql = find_executable_sql_prefix(prefix_text)
#                         if partial_sql is None:
#                             continue

#                         b.num_egd_checks += 1
#                         exec_res = execute_sql_fn(db_id, partial_sql)
#                         if not exec_res.get("ok", False):
#                             # Mark beam as dead; it will be pruned out soon
#                             b.num_egd_failures += 1
#                             b.killed = True
#                             b.logprob = -1e9

#             # Early stop if all beams are finished or killed
#             if all(b.finished or b.killed for b in beams):
#                 break

#         # End of decoding: choose best beam among non-killed
#         alive_beams = [b for b in beams if not b.killed]
#         if alive_beams:
#             best = max(alive_beams, key=lambda b: b.logprob)
#         else:
#             # Fallback: if all killed, pick the least-bad beam overall
#             best = max(beams, key=lambda b: b.logprob)

#         # Decode final text + extract final SQL
#         best_text = tokenizer.decode(best.input_ids, skip_special_tokens=True)
#         best_sql = extract_sql_fn(best_text)

#         # Build debug info
#         beam_debug: List[MidEGDBeamResult] = []
#         for b in beams:
#             txt = tokenizer.decode(b.input_ids, skip_special_tokens=True)
#             sql = extract_sql_fn(txt)
#             beam_debug.append(
#                 MidEGDBeamResult(
#                     text=txt,
#                     sql=sql,
#                     logprob=b.logprob,
#                     killed=b.killed,
#                     num_egd_checks=b.num_egd_checks,
#                     num_egd_failures=b.num_egd_failures,
#                 )
#             )

#         return best_sql, beam_debug




# ************************************
# EGD + GBNF SUPPORT
# ************************************

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class MidEGDConfig:
    num_beams: int = 4
    max_new_tokens: int = 160
    egd_interval: int = 16          # how often (in steps) to run EGD
    min_tokens_for_egd: int = 32    # minimum new tokens before EGD starts
    egd_top_k: int = 2              # how many of the best beams to EGD-check each time


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
    Given the decoded text of a beam, try to extract an *executable* SQL prefix.

    For mid-EGD, we don't require a trailing ';' – for Spider/SQLite it's fine to
    execute a single SELECT without it. We just grab from the first 'select ' onward
    and optionally do a cheap sanity check.
    """
    lower = text.lower()
    first_select = lower.find("select ")
    if first_select == -1:
        return None

    candidate = text[first_select:].strip()

    # Optional: very cheap paren sanity check; you can comment this out if you want
    # SQLite to be the only judge of validity.
    if candidate.count("(") != candidate.count(")"):
        return None

    return candidate


def generate_sql_mid_egd(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    db_id: str,
    execute_sql_fn: Callable[[str, str], Dict[str, Any]],
    extract_sql_fn: Callable[[str], str],
    config: Optional[MidEGDConfig] = None,
    prefix_allowed_tokens_fn: Optional[
        Callable[[int, torch.Tensor], List[int]]
    ] = None,
) -> Tuple[str, List[MidEGDBeamResult]]:
    """
    Manual beam search with mid-decoding Execution-Guided Decoding (EGD),
    optionally constrained by a GBNF prefix_allowed_tokens_fn.

    Critically, this function:
      • Uses the full prompt tokens as the *prefix* for generation.
      • Tracks prompt_len so that *only the generated continuation* is
        decoded and passed to extract_sql_fn and SQLite.

    Args
    ----
    model:
        AutoModelForCausalLM (SQLCoder).
    tokenizer:
        Matching tokenizer.
    prompt:
        Full text prompt.
    db_id:
        Spider db_id used by execute_sql_fn.
    execute_sql_fn:
        (db_id, sql) -> {'ok': bool, 'rows': [...]}
    extract_sql_fn:
        text -> final SQL (for full generations). Called on the *completion* only.
    config:
        MidEGDConfig (beam width, EGD frequency, etc.).
    prefix_allowed_tokens_fn:
        Optional grammar callback with signature (batch_id, input_ids) -> [allowed_ids].

    Returns
    -------
    best_sql: str
        SQL for the best beam after mid-EGD + final exec reranking.
    beam_debug: List[MidEGDBeamResult]
        Debug info for final beams (decoding only the completion).
    """
    if config is None:
        config = MidEGDConfig()

    device = _get_device(model)
    model.eval()

    with torch.no_grad():
        # Encode prompt once
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)[0]  # shape: (prompt_len,)
        prompt_len = input_ids.size(0)
        eos_id = tokenizer.eos_token_id

        # Start with a single beam (prompt only)
        beams: List[BeamState] = [
            BeamState(input_ids=input_ids.clone(), logprob=0.0)
        ]

        # Main decode loop (new tokens)
        for step in range(config.max_new_tokens):
            candidate_beams: List[BeamState] = []

            for beam in beams:
                # If beam is finished or killed, just carry it forward
                if beam.finished or beam.killed:
                    candidate_beams.append(beam)
                    continue

                # Forward pass for this beam
                outputs = model(
                    input_ids=beam.input_ids.unsqueeze(0),  # (1, seq_len)
                    use_cache=False,
                )
                logits = outputs.logits[0, -1, :]  # last token logits, shape: (vocab,)

                # Apply grammar constraints if provided
                if prefix_allowed_tokens_fn is not None:
                    # LMFE's TransformersPrefixAllowedTokensFn expects
                    # a 1D CPU LongTensor: input_ids[batch_id]
                    if isinstance(beam.input_ids, torch.Tensor):
                        ids_for_prefix = beam.input_ids.to("cpu")  # [seq_len]
                    else:
                        ids_for_prefix = torch.tensor(
                            beam.input_ids, dtype=torch.long
                        )

                    allowed_ids = prefix_allowed_tokens_fn(0, ids_for_prefix)

                    # Mask all disallowed tokens in the current logits
                    mask = torch.ones_like(logits, dtype=torch.bool)
                    if len(allowed_ids) > 0:
                        mask[allowed_ids] = False

                    neg_inf = torch.finfo(logits.dtype).min
                    logits = logits.masked_fill(mask, neg_inf)

                logprobs = F.log_softmax(logits, dim=-1)

                # Expand this beam with top-K next tokens
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

            # Prune back to beam width
            candidate_beams.sort(key=lambda b: b.logprob, reverse=True)
            beams = candidate_beams[: config.num_beams]

            # Mid-decoding EGD
            new_tokens_generated = step + 1
            if (
                new_tokens_generated >= config.min_tokens_for_egd
                and new_tokens_generated % config.egd_interval == 0
            ):
                alive_beams = [b for b in beams if not b.killed]
                if alive_beams:
                    # Check top-K alive beams
                    alive_beams.sort(key=lambda b: b.logprob, reverse=True)
                    egd_targets = alive_beams[: config.egd_top_k]

                    for b in egd_targets:
                        # Decode only the *generated continuation*, not the prompt
                        completion_text = tokenizer.decode(
                            b.input_ids[prompt_len:],  # slice off prompt
                            skip_special_tokens=True,
                        )
                        partial_sql = find_executable_sql_prefix(completion_text)
                        if partial_sql is None:
                            continue

                        b.num_egd_checks += 1
                        exec_res = execute_sql_fn(db_id, partial_sql)
                        if not exec_res.get("ok", False):
                            b.num_egd_failures += 1
                            b.killed = True
                            b.logprob = -1e9  # ensure it gets pruned

            # Stop if all beams are finished or killed
            if all(b.finished or b.killed for b in beams):
                break

        # -------------------------
        # Final execution-guided rerank over surviving beams
        # -------------------------
        alive_beams = [b for b in beams if not b.killed]
        if not alive_beams:
            alive_beams = beams  # fallback

        scored: List[
            Tuple[float, float, BeamState, str, str]
        ] = []  # (exec_score, logprob, beam, completion_text, sql)

        for b in alive_beams:
            # Decode only the completion (post-prompt)
            completion_text = tokenizer.decode(
                b.input_ids[prompt_len:],
                skip_special_tokens=True,
            )
            sql = extract_sql_fn(completion_text)
            exec_res = execute_sql_fn(db_id, sql)

            # Cheap execution score: 0.0 = error, 0.5 = ok+empty, 1.0 = ok+non-empty
            if not exec_res.get("ok", False):
                escore = 0.0
            else:
                rows = exec_res.get("rows", [])
                escore = 1.0 if rows else 0.5

            scored.append((escore, b.logprob, b, completion_text, sql))

        # Prefer higher execution score, then higher logprob
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_escore, _, best_beam, best_completion, best_sql = scored[0]

        # Build debug info (again, only decoding the completion)
        beam_debug: List[MidEGDBeamResult] = []
        for b in beams:
            completion_text = tokenizer.decode(
                b.input_ids[prompt_len:],
                skip_special_tokens=True,
            )
            sql = extract_sql_fn(completion_text)
            beam_debug.append(
                MidEGDBeamResult(
                    text=completion_text,
                    sql=sql,
                    logprob=b.logprob,
                    killed=b.killed,
                    num_egd_checks=b.num_egd_checks,
                    num_egd_failures=b.num_egd_failures,
                )
            )

        return best_sql, beam_debug