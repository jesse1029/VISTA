"""Verify recovered original VISTA records. No fitting, tuning, or remote jobs."""
from pathlib import Path
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
import argparse,csv,hashlib,io,json,pickle,sys
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir",type=Path,required=True)
parser.add_argument("--out",type=Path,required=True)
args=parser.parse_args()
DATA=args.data_dir.resolve()
import numpy as np
if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8")
R=ROOT/"records/results"
R4=R/"vista_smst_r4_20260812"
R8=R/"vista_smst_r8_external_2020_20260812_attempt3"
DER=args.out.resolve();DER.mkdir(parents=True,exist_ok=True)
def readcsv(p):
 with p.open(encoding="utf-8-sig",newline="") as h:
  reader=csv.reader(h);original=next(reader);headers=[];seen=set()
  for name in original:
   candidate=name;count=0
   while candidate in seen:
    count+=1;candidate=name+"."+str(count)
   headers.append(candidate);seen.add(candidate)
  return [dict(zip(headers,row)) for row in reader]
def load(p):return json.loads(p.read_text(encoding="utf-8"))
def save(n,d):(DER/n).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class StoredEstimator:
 def __setstate__(self,state):self.__dict__.update(state)
class RestrictedStateReader(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={("numpy","ndarray"):np.ndarray,("numpy","dtype"):np.dtype,
   ("numpy.core.multiarray","_reconstruct"):np.core.multiarray._reconstruct,
   ("numpy._core.multiarray","_reconstruct"):np.core.multiarray._reconstruct,
   ("numpy.core.multiarray","scalar"):np.core.multiarray.scalar,
   ("numpy._core.multiarray","scalar"):np.core.multiarray.scalar,
   ("numpy.core.numeric","_frombuffer"):np.core.numeric._frombuffer,
   ("numpy._core.numeric","_frombuffer"):np.core.numeric._frombuffer,
   ("sklearn.preprocessing._data","StandardScaler"):StoredEstimator,
   ("sklearn.linear_model._logistic","LogisticRegression"):StoredEstimator}
  if (module,name) not in allowed:raise RuntimeError("Unrecognized pickle global: "+module+"."+name)
  return allowed[(module,name)]
model_file=R4/"development/development_models.pkl"
audit=load(R4/"audit/AUDIT_SMST_R4.json")
assert sha(model_file)==audit["bindings"]["development_models"]
assert sha(R4/"run_smst_r4.py")==audit["bindings"]["runner"]
assert sha(R4/"validation/FINAL_DECISION_LOCK.json")==audit["bindings"]["decision"]
models=RestrictedStateReader(io.BytesIO(model_file.read_bytes())).load()
assert len(models)==48
manifest=load(R4/"development/DEVELOPMENT_MANIFEST.json")
decision=load(R4/"validation/FINAL_DECISION_LOCK.json")
b=models[decision["selected"]["model_key"]]
assert b["config"]==decision["selected"]["config"]
classes=b["classes"]
with np.load(DER.parent/"principal_inputs.npz",allow_pickle=False) as z:
 X=z["prisma_common59"];Y=z["desis_reference"];waves=z["wavelength_nm"];ids=z["loc_id"]
raw2021=readcsv(DATA/"desis_2021.csv")
lon=np.array([float(r["longitude"]) for r in raw2021]);lat=np.array([float(r["latitude"]) for r in raw2021])
labels=np.array([str(r["crop_name"]) for r in raw2021])
cells=np.array([f"{int(np.floor((x+180)/.01))}:{int(np.floor((y+90)/.01))}" for x,y in zip(lon,lat)])
buckets=np.array([int.from_bytes(hashlib.sha256(("smst-r4-20260812:"+str(c)).encode("utf-8")).digest()[:8],"big")%10 for c in cells])
roles=np.where(buckets<=5,"development",np.where(buckets<=7,"validation","final"))
partition_report={}
for role in ("development","validation","final"):
 idx=np.flatnonzero(roles==role)
 id_hash=hashlib.sha256("\n".join(str(x) for x in sorted(ids[idx].tolist())).encode()).hexdigest()
 assert id_hash==manifest["split"]["loc_id_list_sha256"][role],role
 counts={str(c):int(np.sum(labels[idx]==c)) for c in classes}
 assert counts==manifest["split"]["counts"][role]
 partition_report[role]={"rows":len(idx),"loc_id_list_sha256":id_hash,"class_counts":counts}
with (DER/"single_split_2021_manifest_regenerated.csv").open("w",encoding="utf-8",newline="") as h:
 w=csv.writer(h);w.writerow(["row_index","loc_id","crop","longitude","latitude","cell","bucket","role"])
 for i in range(len(ids)):w.writerow([i,int(ids[i]),labels[i],lon[i],lat[i],cells[i],int(buckets[i]),roles[i]])
def logits(bundle,S):
 return ((S-bundle["scaler"].mean_)/bundle["scaler"].scale_)@bundle["decoder"].coef_.T+bundle["decoder"].intercept_
def softmax(s):
 p=np.exp(s-s.max(axis=1,keepdims=True));return p/p.sum(axis=1,keepdims=True)
def top_margin(s):
 order=np.argsort(s,axis=1);ii=np.arange(len(s))
 return order[:,-1],s[ii,order[:,-1]]-s[ii,order[:,-2]]
def transport(bundle,z):
 g=z*bundle["global_a"]+bundle["global_b"]
 pg=softmax(logits(bundle,g))
 c=g+np.einsum("nc,ncd->nd",pg,z[:,None,:]*bundle["residual_a"][None,:,:]+bundle["residual_b"][None,:,:])
 fg,mg=top_margin(logits(bundle,g));fc,mc=top_margin(logits(bundle,c))
 keep=(fg==fc)&(mc>=mg+decision["selected"]["margin"])
 return g,c,np.where(keep[:,None],c,g),fg,fc,mg,mc,keep
def rmse(x,y):return np.sqrt(np.mean((x-y)**2,axis=1))
def sam(x,y):
 den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
 cos=np.divide((x*y).sum(axis=1),den,out=np.ones_like(den),where=den>0)
 return np.arccos(np.clip(cos,-1,1))
def macro_f1(true,pred):
 scores=[]
 for cl in classes:
  tp=int(np.sum((true==cl)&(pred==cl)));fp=int(np.sum((true!=cl)&(pred==cl)));fn=int(np.sum((true==cl)&(pred!=cl)))
  scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return sum(scores)/len(scores)
val_idx=np.flatnonzero(roles=="validation")
grids=readcsv(R4/"validation/validation_grid.csv")
assert len(grids)==192 and len({r["model_key"] for r in grids})==48
grid_errors=[];ordering=[]
for key,bundle in models.items():
 g,c,_,fg,fc,mg,mc,_=transport(bundle,X[val_idx])
 f1=macro_f1(labels[val_idx],classes[fg])
 for row in [r for r in grids if r["model_key"]==key]:
  margin=float(row["margin"]);keep=(fg==fc)&(mc>=mg+margin)
  o=np.where(keep[:,None],c,g)
  metric={"coverage":float(keep.mean()),"macro_f1":f1,"rmse":float(rmse(o,Y[val_idx]).mean()),"sam_rad":float(sam(o,Y[val_idx]).mean()),"semantic_label_change_rate_before_gate":float(np.mean(fc!=fg))}
  for k,v in metric.items():
   error=abs(v-float(row[k]));assert error<1e-9,(key,margin,k,error)
   grid_errors.append(error)
  cfg=bundle["config"]
  rank=(-float(row["macro_f1"]),float(row["rmse"]),-float(row["coverage"]),cfg["global_alpha"],cfg["decoder_alpha"],cfg["residual_alpha"],margin)
  ordering.append((rank,key,margin))
ordering.sort()
assert ordering[0][1:]==(decision["selected"]["model_key"],decision["selected"]["margin"])
save("single_split_identity_and_validation.json",{"split":partition_report,"model_count":48,"operating_points":192,"recorded_winner_verified":True,"max_validation_metric_difference":max(grid_errors),"selected":decision["selected"],"original_model_sha256":sha(model_file),"original_runner_sha256":sha(R4/"run_smst_r4.py"),"fits_run":False})
np.savez_compressed(DER/"original_single_split_selected_state.npz",global_a=b["global_a"],global_b=b["global_b"],residual_a=b["residual_a"],residual_b=b["residual_b"],decoder_mean=b["scaler"].mean_,decoder_scale=b["scaler"].scale_,decoder_coef=b["decoder"].coef_,decoder_intercept=b["decoder"].intercept_,classes=classes,wavelength_nm=waves,margin=np.array(decision["selected"]["margin"]))
def evaluate(name,Xe,Ye,ids_e,labels_e,cell_e,source_rows):
 g,c,o,fg,fc,mg,mc,keep=transport(b,Xe)
 rg,rc,ro=rmse(g,Ye),rmse(c,Ye),rmse(o,Ye)
 sg,sc,so=sam(g,Ye),sam(c,Ye),sam(o,Ye)
 top=fg==fc;ref=top_margin(logits(b,Ye))[0]
 source={int(r["loc_id"]):r for r in source_rows}
 assert len(source)==len(ids_e)
 errors=[]
 for i,loc in enumerate(ids_e):
  r=source[int(loc)]
  assert str(classes[fg[i]])==r["global_prediction"] and str(classes[fc[i]])==r["semantic_prediction"]
  global_field="global_rmse"
  output_field="smst_rmse" if name=="single_split_2021" else "smst_top_label_and_raw_margin_rmse"
  accepted_field="smst_committed_semantic" if name=="single_split_2021" else "smst_top_label_and_raw_margin_accept"
  assert bool(keep[i])==(r[accepted_field]=="True")
  for v,k in [(rg[i],global_field),(ro[i],output_field),(mg[i],"global_margin"),(mc[i],"semantic_margin")]:
   err=abs(float(v)-float(r[k]));errors.append(err);assert err<1e-8,(name,loc,k,err)
  if name!="single_split_2021":
   assert bool(top[i])==(r["top_label_only_accept"]=="True")
   err=abs(rc[i]-float(r["always_residual_rmse"]));errors.append(err);assert err<1e-8
   assert str(cell_e[i])==r["spatial_cell"]
 groups={}
 for title,mask in [("all",np.ones(len(ids_e),dtype=bool)),("top_label",top),("retained",keep),("pruned",top&~keep)]:
  groups[title]={"rows":int(mask.sum()),"cells":len(set(cell_e[mask])),"mean_candidate_delta_rmse":float(np.mean((rc-rg)[mask])),"mean_candidate_delta_sam":float(np.mean((sc-sg)[mask])),"candidate_tail_events":int(np.sum(((rc-rg)>.25)&(((rc-rg)/np.maximum(rg,1e-12))>.05)&mask))}
 with (DER/(name+"_per_location_verified.csv")).open("w",encoding="utf-8",newline="") as h:
  w=csv.writer(h);w.writerow(["loc_id","crop","spatial_cell","reference_label","fallback_label","candidate_label","top_label_compatible","retained","pruned","fallback_rmse","candidate_rmse","returned_rmse","candidate_delta_rmse","candidate_delta_sam","candidate_tail_event"]+[f"fallback_logit_{j}" for j in range(5)]+[f"candidate_logit_{j}" for j in range(5)])
  lg=logits(b,g);lc=logits(b,c)
  for i in range(len(ids_e)):
   dr=rc[i]-rg[i]
   w.writerow([int(ids_e[i]),labels_e[i],cell_e[i],classes[ref[i]],classes[fg[i]],classes[fc[i]],int(top[i]),int(keep[i]),int(top[i] and not keep[i]),rg[i],rc[i],ro[i],dr,sc[i]-sg[i],int(dr>.25 and dr/max(rg[i],1e-12)>.05)]+lg[i].tolist()+lc[i].tolist())
 np.savez_compressed(DER/(name+"_spectra_from_original_state.npz"),loc_id=ids_e,classes=classes,wavelength_nm=waves,prisma_common59=Xe,desis_reference=Ye,fallback=g,candidate=c,returned=o,fallback_logits=logits(b,g),candidate_logits=logits(b,c),reference_label=ref,top_label=top,retained=keep,spatial_cell=cell_e)
 changed=fc!=fg
 anchor={"reference_accuracy":float(np.mean(classes[ref]==labels_e)),"reference_macro_f1":macro_f1(labels_e,classes[ref]),"fallback_agree_with_reference":float(np.mean(fg==ref)),"changes":int(changed.sum()),"toward":int(np.sum(changed&(fc==ref)&(fg!=ref))),"away":int(np.sum(changed&(fg==ref)&(fc!=ref))),"between":int(np.sum(changed&(fg!=ref)&(fc!=ref)))}
 return {"groups":groups,"fallback_mean_rmse":float(rg.mean()),"returned_mean_rmse":float(ro.mean()),"returned_mean_sam":float(so.mean()),"mean_returned_delta_rmse":float(np.mean(ro-rg)),"reference_anchor":anchor,"maximum_difference_from_original_csv":max(errors)}
fi=np.flatnonzero(roles=="final")
report21=evaluate("single_split_2021",X[fi],Y[fi],ids[fi],labels[fi],cells[fi],readcsv(R4/"final/final_per_location.csv"))
expected=load(R4/"final/FINAL_SUMMARY.json")
assert abs(report21["returned_mean_rmse"]-expected["metrics"]["smst"]["rmse"])<1e-10
raw=DATA
summary8=load(R8/"R8_EXTERNAL_SUMMARY.json")
for file in ("desis_2020.csv","prisma_2020.csv","metadata_2020.xml"):
 assert sha(raw/file)==summary8["input_sha256"][file]
dd=[r for r in readcsv(raw/"desis_2020.csv") if r["Image ID"]==summary8["external_pair"]["desis_image"]]
pp=readcsv(raw/"prisma_2020.csv")
assert len(dd)==len(pp)==1490
d_by_id={int(r["Unique ID"]):r for r in dd}
dd=[d_by_id[int(r["Unique ID"])] for r in pp]
assert all(float(d["Longitude"])==float(p["Longitude"]) and float(d["Latitude"])==float(p["Latitude"]) for d,p in zip(dd,pp))
dc=[k for k in dd[0] if k.startswith("X") and k[1:].isdigit()]
pc=[k for k in pp[0] if k.startswith("X") and k[1:].isdigit()]
dw=np.array([int(k[1:]) for k in dc]);pw=np.array([int(k[1:]) for k in pc])
Y20=np.stack([np.interp(waves,dw,np.array([float(r[k]) for k in dc])) for r in dd])
X20=np.stack([np.interp(waves,pw,np.array([float(r[k]) for k in pc])) for r in pp])
ids20=np.array([int(r["Unique ID"]) for r in pp]);labs20=np.array([r["Crop Type"] for r in pp])
cells20=np.array([f"{int(np.floor((float(r['Longitude'])+180)/.01))}:{int(np.floor((float(r['Latitude'])+90)/.01))}" for r in pp])
report20=evaluate("frozen_2020",X20,Y20,ids20,labs20,cells20,readcsv(R8/"R8_PER_LOCATION.csv"))
assert report20["groups"]["all"]["cells"]==238
assert report20["groups"]["top_label"]["cells"]==216
assert (report20["groups"]["retained"]["rows"],report20["groups"]["pruned"]["rows"])==(450,531)
save("original_state_replay_and_cells.json",{"completed_utc":datetime.now(timezone.utc).isoformat(),"single_split_2021":report21,"frozen_2020":report20,"cell_count_explanation":"238 cells cover all1490 frozen2020 rows;216 cells cover981 top-label-compatible rows before retained/pruned split.","new_fitting":False,"original_source_or_model_changed":False,"protocol_verification":"Original R8 search/temperature/tie rules and S10 bootstrap are recovered and verified in R8_selector_search_and_bootstrap_verified.json. The manuscript S11 calculation uses 5000 draws and the explicitly declared seed 20261018."})
print(json.dumps({"single_split_2021":report21,"frozen_2020":report20,"validation_operating_points_verified":192,"validation_max_difference":max(grid_errors)},ensure_ascii=False,indent=2))

