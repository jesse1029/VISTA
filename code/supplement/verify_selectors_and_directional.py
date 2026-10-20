"""Read-only checks of original R8 selectors, bootstrap and R5 stored outputs."""
from pathlib import Path
import contextlib,io,importlib.util,json,csv,hashlib,sys
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("historical_replay",ROOT/"verify_historical_records.py")
h=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):spec.loader.exec_module(h)
np=h.np
s=h.b
source=ROOT/"records/experiments/smst_r8_external_2020/run_smst_r8_external_2020.py"
audit=h.load(h.R8/"R8_AUDIT.json")
assert h.sha(source)==audit["r8_runner_sha256"]
prereg=source.parent/"PREREGISTRATION.md"
assert h.sha(prereg)==audit["preregistration_sha256"]
summary=h.load(h.R8/"R8_EXTERNAL_SUMMARY.json")
g,c,o,fg,fc,mg,mc,keep=h.transport(s,h.X[h.val_idx])
val_scores=h.logits(s,c)
y_idx=np.array([np.flatnonzero(h.classes==v)[0] for v in h.labels[h.val_idx]])
temps=[]
for t in (.25,.5,1.,2.,4.):
 p=h.softmax(val_scores/t)
 nll=float(-np.log(np.clip(p[np.arange(len(p)),y_idx],1e-12,1)).mean())
 temps.append({"temperature":t,"validation_nll":nll})
chosen=min(temps,key=lambda r:(r["validation_nll"],r["temperature"]))["temperature"]
assert chosen==summary["selector_lock"]["temperature"]
same=fg==fc
assert int(same.sum())==summary["selector_lock"]["validation_top_label_count"]
assert int(keep.sum())==summary["selector_lock"]["validation_both_count"]
fraction=float(keep.sum()/same.sum())
assert fraction==summary["selector_lock"]["validation_conditional_acceptance_fraction"]
g,c,o,fg,fc,mg,mc,keep=h.transport(s,h.X20)
scores=h.logits(s,c)
p=h.softmax(scores/chosen)
confidence=p.max(axis=1)
entropy=-(p*np.log(np.clip(p,1e-12,1))).sum(axis=1)
norm_margin=mc/np.maximum(np.linalg.norm(scores,axis=1),1e-12)
same=fg==fc
count=int(round(fraction*int(same.sum())))
assert count==778
def ranked(score):
 mask=np.zeros(len(same),dtype=bool)
 candidates=np.flatnonzero(same)
 order=np.lexsort((h.ids20[candidates],-np.nan_to_num(score[candidates],nan=-np.inf)))
 mask[candidates[order[:count]]]=True
 return mask
selectors={"always_residual":np.ones(len(same),dtype=bool),"top_label_only":same,"margin_only":mc>=mg,"smst_top_label_and_raw_margin":keep,"top_label_temperature_probability_matched":ranked(confidence),"top_label_entropy_matched":ranked(-entropy),"top_label_normalized_margin_matched":ranked(norm_margin)}
rows20=h.readcsv(h.R8/"R8_PER_LOCATION.csv")
cells,inverse=np.unique(h.cells20,return_inverse=True)
group=[np.flatnonzero(inverse==i) for i in range(len(cells))]
rng=np.random.default_rng(20260812)
draws=rng.integers(0,len(cells),size=(1000,len(cells)))
replicate_rows=[np.concatenate([group[i] for i in selected]) for selected in draws]
np.savez_compressed(h.DER/"S10_original_seed_bootstrap_draws_regenerated.npz",seed=np.array(20260812),spatial_cells=cells,draw_indices=draws)
rg=h.rmse(g,h.Y20);sg=h.sam(g,h.Y20)
checks=[]
for name,mask in selectors.items():
 assert np.array_equal(mask,np.array([r[name+"_accept"]=="True" for r in rows20])),name
 output=np.where(mask[:,None],c,g)
 rmse=h.rmse(output,h.Y20);sam=h.sam(output,h.Y20)
 dr=rmse-rg;ds=sam-sg
 tail=(dr>.25)&((dr/np.maximum(rg,1e-12))>.05)
 orig=summary["selectors_external"][name]
 assert int(mask.sum())==orig["accepted_rows"]
 assert int(tail.sum())==orig["declared_harms_vs_global"]
 assert abs(float(rmse.mean())-orig["spectral_rmse"])<1e-10
 assert abs(float(sam.mean())-orig["sam_rad"])<1e-10
 values=np.array([[dr[rr].mean(),ds[rr].mean(),mask[rr].mean(),tail[rr].mean()] for rr in replicate_rows])
 err=0.
 for j,key in enumerate(("delta_rmse","delta_sam_rad","coverage","harm_rate_all")):
  expected=orig["spatial_cell_bootstrap"][key]
  err=max(err,abs(float(values[:,j].mean())-expected["mean"]),float(np.max(np.abs(np.quantile(values[:,j],[.025,.975])-np.array(expected["ci95"])))))
 assert err<1e-10,(name,err)
 checks.append({"selector":name,"accepted_rows":int(mask.sum()),"events":int(tail.sum()),"all_row_memberships_match":True,"max_bootstrap_json_difference":err})
h.save("R8_selector_search_and_bootstrap_verified.json",{"original_runner_sha256":h.sha(source),"original_preregistration_sha256":h.sha(prereg),"temperature_grid":[.25,.5,1.,2.,4.],"temperature_scores_regenerated_from_original_state":temps,"selected_temperature":chosen,"temperature_tie_rule":"Minimum(validation NLL,temperature)","rank_tie_rule":"Descending score then ascending loc_id; NumPy lexsort","matched_count_rule":"int(round(validation_L_and_M/validation_L * external_L_count))","matched_count":count,"baseline_grids":{"ridge_alpha":[1e-6,1e-4,1e-2,1.0],"pls_components":[2,4,8,16,32],"pds_half_width":[1,3,5],"cluster_count":[2,3,5,7]},"baseline_selection_rule":"Minimum validation RMSE, then configuration-key tuple in source; no parameter-count tiebreak","baseline_all_candidate_scores_or_models_resaved":False,"bootstrap_scope":"Original S10 all238cells;1000draws,seed20260812. S11 uses 5000 draws with its separately declared seed 20261018.","selectors":checks,"fit_calls":False})
R5=h.R/"vista_smst_r5_regional_20260812"
r5=h.load(R5/"run/R5_SUMMARY.json")
rows5=h.readcsv(R5/"run/R5_PER_LOCATION.csv")
geo=r5["geography"];west=h.lon<=geo["longitude_threshold"]-geo["gap_degrees"];east=h.lon>=geo["longitude_threshold"]+geo["gap_degrees"]
assert int(west.sum())==geo["west_rows"] and int(east.sum())==geo["east_rows"]
assert int((~(west|east)).sum())==geo["excluded_rows"]
with (h.DER/"directional_geographic_membership_regenerated.csv").open("w",encoding="utf-8",newline="") as f:
 w=csv.writer(f);w.writerow(["row_index","loc_id","longitude","latitude","side"])
 for i in range(len(h.ids)):w.writerow([i,int(h.ids[i]),h.lon[i],h.lat[i],"west" if west[i] else "east" if east[i] else "excluded"])
reports5=[]
for direction,test in [("west_to_east",east),("east_to_west",west)]:
 rr=[r for r in rows5 if r["direction"]==direction]
 assert set(int(r["loc_id"]) for r in rr)==set(h.ids[test].tolist())
 expected=r5["directions"][direction]
 for column,metric in [("global_rmse","global_ridge"),("smst_rmse","smst")]:
  mean=sum(float(r[column]) for r in rr)/len(rr)
  assert abs(mean-expected["metrics"][metric]["rmse"])<1e-10
 reports5.append({"direction":direction,"rows":len(rr),"membership_matches_original":True,"metrics_match":True})
h.save("R5_geographic_records_verified.json",{"geometry":geo,"checks":reports5,"source_audit_sha256":h.sha(R5/"audit/AUDIT_SMST_R5_REGIONAL.json"),"original_directional_models_not_serialized_by_runner":True,"fitting_run":False})
print(json.dumps({"R8_selectors_checked":len(checks),"S10_bootstrap_draws":1000,"S10_bootstrap_seed":20260812,"max_bootstrap_difference":max(r["max_bootstrap_json_difference"] for r in checks),"R5_directional_rows_verified":len(rows5),"temperature_grid":temps,"all_checks":"PASS"},ensure_ascii=False,indent=2))

