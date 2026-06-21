"""Integration: the Colab paste-back handback path (Modules 7-9).

A student trains on Colab and pastes a one-line submission token into the
off-Colab grader. We run the answer-key training under the CPU config (pinned
seeds, no GPU needed in CI), build the grading policy from that reference run,
then assert the genuine token is accepted and tampered tokens are rejected for
the right reason. Loss is graded by a tolerance band, never bit-exact.
"""

import torch

from reference.gpt.model import GPT
from reference.gpt.train import train, make_submission_token, arch_signature, sample_hash
from grader.submission_token import parse_token, grade, Policy

DATA = "to be or not to be that is the question " * 30


def _reference_run():
    res = train(config="cpu", max_iters=20, seed=0, data=DATA)
    token = make_submission_token(res, mod="M8")
    fields = parse_token(token)
    # Policy: loss band sits above the reference final loss; arch + sample exact.
    policy = Policy(loss_max=fields["loss"] + 0.5, arch=fields["arch"], shash=fields["shash"], mod="M8")
    return res, token, policy


def test_genuine_token_is_accepted():
    _, token, policy = _reference_run()
    assert grade(token, policy).passed


def test_loss_above_band_is_rejected():
    _, _, policy = _reference_run()
    bad = f"SLM-M8 loss={policy.loss_max + 5:.4f} arch={policy.arch} shash={policy.shash}"
    r = grade(bad, policy)
    assert not r.passed and "loss" in r.failed_checks


def test_swapped_sample_hash_is_rejected():
    _, _, policy = _reference_run()
    bad = f"SLM-M8 loss=0.0001 arch={policy.arch} shash=deadbeef"
    r = grade(bad, policy)
    assert not r.passed and "shash" in r.failed_checks


def test_truncated_token_fails_to_parse():
    _, _, policy = _reference_run()
    r = grade("SLM-M8 loss=1.0 arch=abc", policy)
    assert not r.passed and "parse" in r.failed_checks


def test_architecture_signature_is_reproducible():
    # Same seed -> identical architecture signature (structure is value-independent).
    a = train(config="cpu", max_iters=5, seed=0, data=DATA)
    b = train(config="cpu", max_iters=5, seed=0, data=DATA)
    assert arch_signature(a["model"]) == arch_signature(b["model"])


def test_checkpoint_roundtrips_to_identical_logits():
    res = train(config="cpu", max_iters=10, seed=0, data=DATA)
    model = res["model"].eval()
    x = torch.zeros(1, 8, dtype=torch.long)
    with torch.no_grad():
        logits0, _ = model(x)
    reloaded = GPT(res["config"])
    reloaded.load_state_dict(model.state_dict())
    reloaded.eval()
    with torch.no_grad():
        logits1, _ = reloaded(x)
    assert torch.allclose(logits0, logits1, atol=1e-6)
