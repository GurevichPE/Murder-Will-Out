import numpy as np
import math
import torch



def logprob_of_sequence(question: str, answer: str):
    """
    Returns (log_prob, length_normalized_log_prob)
    log_prob = sum_i log P(a_i | q, a_<i)
    length_normalized = log_prob / n_tokens_answer
    """
    # Form text as prompt + space + answer (match how you generate)
    text = (question.strip() + " " + answer.strip()).strip()
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        # Get logits for each position
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is average negative log-likelihood per token; multiply back to get sum NLL
        avg_nll = outputs.loss.item()  # negative log-likelihood per token
        n_tokens = input_ids.size(1)
        sum_logprob = - avg_nll * n_tokens  # sum log-prob over all tokens (incl prompt)
    # We only want the logprob of the answer tokens (not prompt). To compute exactly, it's safer to compute per-token logits:
    # More exact (and recommended): recompute logits, then sum only the answer-token conditionals (see compute_logprob_answer below).
    return sum_logprob, sum_logprob / n_tokens

# Precise variant: compute conditional probabilities for only the answer tokens
def answer_logprob_precise(question: str, answer: str):
    prompt = question.strip() + " "
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"][0]
    answer_ids = tokenizer(answer, return_tensors="pt").to(device)["input_ids"][0]
    # build concatenated input: [prompt_ids, answer_ids]
    input_ids = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0)  # shape 1 x L
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # shape L x V
    # compute log probs for answer token positions
    log_probs = torch.log_softmax(logits, dim=-1)  # (L, V)
    # answer positions start at index len(prompt_ids)
    start = len(prompt_ids)
    sum_logprob = 0.0
    for i, tid in enumerate(answer_ids):
        pos = start + i
        sum_logprob += log_probs[pos-1, tid].item()  # conditional prob P(tok_i | all prev tokens)
        # note: using pos-1 because the model's logits at position t predict token t (shifted)
    n = len(answer_ids)
    return sum_logprob, sum_logprob / n


# verification_score.py
def p_true_given_qa(question: str, answer: str, verifier_prompt_template=None):
    """
    Returns P(True | question, answer) computed as joint prob of tokens in the string "True"
    under the model when prompted with a verification prompt.
    verifier_prompt_template: a template string with placeholders `{q}` and `{a}`.
    """
    if verifier_prompt_template is None:
        verifier_prompt_template = (
            "Question: {q}\n"
            "Answer: {a}\n"
            "Is the proposed answer correct? Answer 'True' or 'False'.\n"
            "Output:"
        )
    prompt = verifier_prompt_template.format(q=question, a=answer)
    # We want the conditional prob of generating " True" or "True" tokens
    true_text = " True"  # include leading space if tokenizer splits that way
    # Encode prompt and the target tokens
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"][0]
    true_ids = tokenizer(true_text, add_special_tokens=False, return_tensors="pt").to(device)["input_ids"][0]
    input_ids = torch.cat([prompt_ids, true_ids], dim=0).unsqueeze(0)  # 1 x L
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0]  # (L, V)
    log_probs = torch.log_softmax(logits, dim=-1)
    # compute joint log-prob of true_ids as conditionals
    start = len(prompt_ids)
    joint_logp = 0.0
    for i, tid in enumerate(true_ids):
        pos = start + i
        # the model's logit at position pos-1 predicts token at pos
        joint_logp += log_probs[pos-1, tid].item()
    p_true = math.exp(joint_logp)
    return p_true


