import re
from reference.gpt.train import train, make_submission_token

DATA = "to be or not to be that is the question " * 30

def test_training_reduces_loss():
    res = train(config="cpu", max_iters=40, seed=0, data=DATA)
    assert res["final_loss"] < res["initial_loss"]

def test_submission_token_format():
    res = train(config="cpu", max_iters=10, seed=0, data=DATA)
    tok = make_submission_token(res, mod="M8")
    assert re.match(r"^SLM-M8 loss=\d+\.\d+ arch=[0-9a-f]+ shash=[0-9a-f]+$", tok)
