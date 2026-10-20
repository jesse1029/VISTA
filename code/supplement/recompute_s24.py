"""Independent recomputation of the S24 shared-crop intervals (joint cell bootstrap)."""
import sys,json,hashlib
import numpy as np, pandas as pd
src,out=sys.argv[1],sys.argv[2]
d=pd.read_csv(src)
crops=['almonds','corn','grapes','pistachios']
s=d[d.crop_label.isin(crops)].copy()
s['dr']=s.smst_top_label_and_raw_margin_rmse-s.global_rmse
cells=sorted(s.spatial_cell.astype(str).unique())
idx={c:np.flatnonzero(s.spatial_cell.astype(str).values==c) for c in cells}
rng=np.random.default_rng(20260812)
dr=s.dr.values; lab=s.crop_label.values
res=[]
for _ in range(1000):
    pick=rng.integers(0,len(cells),len(cells))
    rows=np.concatenate([idx[cells[k]] for k in pick])
    m={c:dr[rows][lab[rows]==c].mean() for c in crops}
    res.append([m[c] for c in crops]+[np.mean([m[c] for c in crops]), dr[rows].mean()])
res=np.array(res)
q=lambda j:[float(np.percentile(res[:,j],2.5)),float(np.percentile(res[:,j],97.5))]
J=dict(source_sha256=hashlib.sha256(open(src,'rb').read()).hexdigest(),n_cells=len(cells),
  crops={c:dict(rows=int((lab==c).sum()),point=float(dr[lab==c].mean()),ci95=q(i)) for i,c in enumerate(crops)},
  equal_crop=dict(point=float(np.mean([dr[lab==c].mean() for c in crops])),ci95=q(4)),
  shared_row_weighted=dict(point=float(dr.mean()),ci95=q(5)))
open(out,'w').write(json.dumps(J,indent=1)); print(json.dumps(J,indent=1))
