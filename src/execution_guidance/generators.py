from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class BeamCandidate:
    """
    Raw beam-level candidate before execution-guided scoring.

    Attributes
    ----------
    text:
        The decoded text for this candidate (usually contains just the SQL
        completion for the prompt, depending on how you decode).
    logprob:
        Sum of log-probabilities of the generated tokens conditioned on the
        prompt (higher is better).
    sequence_ids:
        Full token IDs (prompt + generated tokens) for this candidate. This
        can be useful later if you want to inspect or re-score sequences.
    """
    text: str
    logprob: float
    sequence_ids: torch.LongTensor


@torch.no_grad()
def compute_candidate_logprob(
    model: PreTrainedModel,
    sequence_ids: torch.LongTensor,
    prompt_len: int,
) -> float:
    """
    Compute the log-probability of the generated suffix tokens given the prompt.

    We run a forward pass over the full sequence (prompt + generated), take the
    log-softmax over the vocabulary at each position, and accumulate the
    log-probability assigned to each generated token.

    Parameters
    ----------
    model:
        The underlying causal LM (e.g., SQLCoder).
    sequence_ids:
        1D tensor of token IDs: [prompt_tokens..., generated_tokens...].
    prompt_len:
        Number of tokens coming from the prompt encoding. Only positions
        >= prompt_len are considered "generated" and contribute to the score.
    """
    device = model.device
    seq = sequence_ids.to(device).unsqueeze(0)  # [1, L]
    outputs = model(input_ids=seq)
    logits = outputs.logits  # [1, L, V]
    logprobs = F.log_softmax(logits, dim=-1)

    L = seq.size(1)
    total_logprob = 0.0

    # Generated token at position t is predicted from logits at position t-1.
    for t in range(prompt_len, L):
        token_id = seq[0, t].item()
        prev_index = t - 1
        token_logprob = logprobs[0, prev_index, token_id].item()
        total_logprob += float(token_logprob)

    return float(total_logprob)


@torch.no_grad()
def generate_beam_candidates(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    num_beams: int = 8,
    num_return_sequences: int = 8,
    max_new_tokens: int = 160,
) -> List[BeamCandidate]:
    """
    Run beam search with the underlying SQLCoder model and return raw beam
    candidates (text + logprob), without any execution guidance applied yet.

    We deliberately do not apply any grammar constraints here; this is the
    "cheapest" EGD setup that simply reranks finished beams based on execution.
    """
    device = model.device

    enc = tokenizer(
        prompt,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    prompt_len = input_ids.shape[1]

    # Use standard HF generate to obtain multiple beams. We don't request
    # output_scores here because we recompute logprobs in a second pass using
    # compute_candidate_logprob(), which is simpler and more robust to the
    # internal beam bookkeeping.
    gen_outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=False,
        early_stopping=True,
    )

    # gen_outputs has shape [num_return_sequences, prompt_len + gen_len]
    candidates: List[BeamCandidate] = []
    for seq_ids in gen_outputs:
        seq_ids = seq_ids.to(device)
        text = tokenizer.decode(seq_ids, skip_special_tokens=True)
        lp = compute_candidate_logprob(model, seq_ids, prompt_len=prompt_len)
        candidates.append(
            BeamCandidate(
                text=text,
                logprob=lp,
                sequence_ids=seq_ids.cpu(),
            )
        )

    return candidates
