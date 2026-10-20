"""Reference-spectrum classifier agreement for a rerun (Supplement S5).
Usage: python reference_anchor.py ORIGINAL.py DESIS PRISMA RUN_BASE_DIR OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vista_converged
o = vista_converged.load(Path(sys.argv[1]))
data = o.load_release(Path(sys.argv[2]), Path(sys.argv[3])); run = Path(sys.argv[4])
parts = o.make_partitions(data["lon"], data["lat"]); x, y, lab = data["x"], data["y"], data["class_idx"]
rows = []
for part in parts:
    k, dev, test = part["fold"], part["dev"], part["test"]
    d = json.loads((run / f"fold_{k}_decision.json").read_text()); s1 = d["stage1"]["params"]; lp = d["families"]["logistic_partial"]["params"]
    g = o.fit_diag_affine(x[dev], y[dev], s1["global_alpha"]); dec = o.fit_logistic(y[dev], lab[dev], s1["decoder_alpha"])
    gd, gt = o.apply_diag(x[dev], g), o.apply_diag(x[test], g)
    st = o.fit_weighted_residual(x[dev], y[dev], gd, dec.proba(gd), lp["residual_alpha"])
    full = o.apply_weighted_residual(x[test], gt, dec.proba(gt), st); cand = gt + lp["scale"] * (full - gt)
    ref, fb, cd = dec.predict_idx(y[test]), dec.predict_idx(gt), dec.predict_idx(cand)
    for i, idx in enumerate(test):
        rows.append({"row_index": int(idx), "crop": data["crop"][idx], "class_idx": int(lab[idx]), "ref": int(ref[i]), "fb": int(fb[i]), "cand": int(cd[i])})
f = pd.DataFrame(rows)
def summ(g):
    ch = g.cand != g.fb
    return {"n": len(g), "ref_acc": float(accuracy_score(g.class_idx, g.ref)),
            "ref_f1": float(f1_score(g.class_idx, g.ref, labels=np.arange(5), average="macro", zero_division=0)),
            "agree": float((g.fb == g.ref).mean()), "changes": int(ch.sum()),
            "toward": int((ch & (g.cand == g.ref) & (g.fb != g.ref)).sum()),
            "away": int((ch & (g.fb == g.ref) & (g.cand != g.ref)).sum()),
            "between": int((ch & (g.fb != g.ref) & (g.cand != g.ref)).sum())}
out = {"overall": summ(f), "crop": {c: summ(g) for c, g in f.groupby("crop")}}
json.dump(out, open(sys.argv[5], "w"), indent=1); print(json.dumps(out, indent=1))
