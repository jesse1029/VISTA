"""Recompute the manuscript S11 intervals from fixed outputs, seed 20261018."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
DER=args.out.resolve()
output={"status":"MANUSCRIPT_DECLARED_SEED_20261018","seed":20261018,"replicates":5000,"numpy_version":np.__version__,"draw_rule":"Reset default_rng(seed) per condition; sort occupied L-cell strings lexicographically; sample same number of cells with replacement; retain all L rows in each drawn cell; compute retained-minus-pruned within every draw","historical_seed_recovered":False,"new_model_fitting":False,"conditions":{}}
for name in ("single_split_2021","frozen_2020"):
 src=DER/(name+"_per_location_verified.csv")
 with src.open(encoding="utf-8",newline="") as f:rows=[r for r in csv.DictReader(f) if r["top_label_compatible"]=="1"]
 cells=np.array(sorted({r["spatial_cell"] for r in rows}));lookup={c:i for i,c in enumerate(cells)}
 stats=np.zeros((len(cells),6))
 for r in rows:
  i=lookup[r["spatial_cell"]];group=0 if r["retained"]=="1" else 1
  stats[i,group]+=1;stats[i,2+group]+=float(r["candidate_delta_rmse"]);stats[i,4+group]+=int(r["candidate_tail_event"])
 rng=np.random.default_rng(20261018)
 draws=rng.integers(0,len(cells),size=(5000,len(cells)))
 sums=stats[draws].sum(axis=1)
 valid=(sums[:,0]>0)&(sums[:,1]>0)
 drmse=sums[valid,2]/sums[valid,0]-sums[valid,3]/sums[valid,1]
 dtail=100*(sums[valid,4]/sums[valid,0]-sums[valid,5]/sums[valid,1])
 total=stats.sum(axis=0)
 dpoint=total[2]/total[0]-total[3]/total[1]
 tpoint=100*(total[4]/total[0]-total[5]/total[1])
 output["conditions"][name]={"rows_in_L":len(rows),"cells_in_L":len(cells),"retained_rows":int(total[0]),"pruned_rows":int(total[1]),"valid_replicates":int(valid.sum()),"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"retained_minus_pruned_delta_rmse":{"point":float(dpoint),"ci95":np.quantile(drmse,[.025,.975]).tolist()},"retained_minus_pruned_tail_rate_percentage_points":{"point":float(tpoint),"ci95":np.quantile(dtail,[.025,.975]).tolist()}}
 with (DER/(name+"_S11_draws.csv")).open("w",encoding="utf-8",newline="") as f:
  w=csv.writer(f);w.writerow(["replicate","delta_rmse_retained_minus_pruned","delta_tail_rate_percentage_points"])
  w.writerows(zip(np.flatnonzero(valid).tolist(),drmse.tolist(),dtail.tolist()))
(DER/"S11_seed20261018.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(output,ensure_ascii=False,indent=2))

