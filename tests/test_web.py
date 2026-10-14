import json
import shutil
import subprocess
from pathlib import Path

import pytest

from suaraulasan import export_web, train

TEXTS = ["aplikasinya mantap, sangat membantu", "sering error, saldo nyangkut terus!!", "ok", "Gak bisa login udah 3 hari"]


@pytest.mark.skipif(not shutil.which("node"), reason="butuh node")
def test_js_matches_sklearn(tmp_path):
    model, _ = train.train(train.load("data/sample.csv"))
    (tmp_path / "m.json").write_text(json.dumps(export_web.export(model)))
    js = f"const p=require('{Path('docs/predict.js').resolve()}'),m=require('{tmp_path}/m.json');console.log(JSON.stringify({json.dumps(TEXTS)}.map(t=>p(m,t).score)))"
    got = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    for g, want in zip(got, model.decision_function(TEXTS)):
        assert abs(g - want) < 0.01
