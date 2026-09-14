#!/usr/bin/env python3
"""Pooled ratios per channel, configuration and cell from the measurement-convention scan,
with patient-cluster bootstrap intervals for the burden ratio. The default configuration
reproduces the held-out numbers and is checked against them.
"""
from alarmreplay.paths import WORK as R, COHORT as SAMP
import csv,glob,json,os,collections,sys
import numpy as np
files=sorted(glob.glob(f"{R}/sensitivity/s2_rows_*.csv")) if not os.environ.get("ALARM_TRIAL") else sorted(glob.glob(f"{R}/cache/_trial_s2_rows_*.csv"))
# ---- completion check: every stripe's state file must report DONE (i == number of admissions in the stripe)
if not os.environ.get("ALARM_TRIAL"):
    splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
    ntest=sum(1 for r in csv.DictReader(open(SAMP)) if splitmap.get(r["aMRN"])=="TEST")
    # The scan may have been run in any number of stripes; the number is the number of state files,
    # and stripe k of N holds the split's admissions with index i % N == k, exactly as
    # scan/scan_measurement_conventions.py allocates them.
    N=int(os.environ.get("ALARM_STRIPES") or len(glob.glob(f"{R}/cache/s2_state_[0-9]*.json")))
    prog=[]; done=N>0
    for k in range(N):
        n_k=len([i for i in range(ntest) if i%N==k]); st=f"{R}/cache/s2_state_{k}.json"
        i=json.load(open(st))["i"] if os.path.exists(st) else 0
        prog.append(f"stripe {k}: {i}/{n_k}"); done&=(i>=n_k)
    print(f"Tier-2 scan progress ({N} stripes):", "; ".join(prog))
    if not done or not files:
        raise SystemExit(f"measurement-convention scan not finished ({N} stripe(s) found): no aggregate written")
agg=collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0]*6)); seen=set()
for fp in files:
    for r in csv.DictReader(open(fp)):
        key=(r["aMRN"],r["aCSN"],r["channel"],r["config"],r["cell"])
        if key in seen: continue
        seen.add(key)
        a=agg[(r["channel"],r["config"],r["cell"])][r["aMRN"]]
        for i,c in enumerate(["rec_runs","match","recon_runs","recon_secs","rec_secs","onset_within"]): a[i]+=float(r[c])
ORDER=["default","d0_8","d0_16","d0_60","tol_10","tol_20","tol_60","onset_6","onset_20","sil_include","lim_modal"]
CHN={"ecgResp-numLimit-respRate-low":"RR low","ecg-numLimit-heartRate-high":"HR high","ecg-numLimit-heartRate-low":"HR low","spO2-numLimit-satO2-low":"SpO2 low"}
rng=np.random.default_rng(20260831); B=1000
out=[]; base={}
for (ch,cfg,cell),per in sorted(agg.items(),key=lambda kv:(list(CHN).index(kv[0][0]),ORDER.index(kv[0][1]),kv[0][2])):
    pats=sorted(per); M=np.array([per[p] for p in pats]); m=M[:,4]>0; M=M[m]
    tot=M.sum(axis=0); n=len(M)
    rho=tot[3]/tot[4]; rec=tot[1]/tot[0] if tot[0] else float("nan"); pre=tot[1]/tot[2] if tot[2] else float("nan"); on=tot[5]/tot[1] if tot[1] else float("nan")
    idx=rng.integers(0,n,size=(B,n)); rr=M[idx,3].sum(axis=1)/M[idx,4].sum(axis=1)
    lo,hi=np.percentile(rr,[2.5,97.5])
    if cfg=="default": base[(ch,cell)]=rho
    out.append(dict(channel=ch,config=cfg,cell=cell,n_patients=int(n),burden_ratio=round(rho,4),ci_lo=round(float(lo),4),ci_hi=round(float(hi),4),
                    recall=round(rec,4),precision=round(pre,4),onset_agreement=round(on,4),recorded_runs=int(tot[0]),recorded_secs=int(tot[4])))
for o in out: o["ratio_to_default"]=round(o["burden_ratio"]/base[(o["channel"],o["cell"])],4) if (o["channel"],o["cell"]) in base else None
os.makedirs(f"{R}/sensitivity",exist_ok=True)
pre="_trial_" if os.environ.get("ALARM_TRIAL") else ""
with open(f"{R}/sensitivity/{pre}S2_CONVENTION_SCAN.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
json.dump(out,open(f"{R}/sensitivity/{pre}S2_CONVENTION_SCAN.json","w"),indent=1)
# LaTeX tables: (A) burden ratios under delta0 / silencing / modal-limit perturbations; (B) recall + onset at the selected cell under tolerance / onset-window perturbations
LAB={"default":"Fixed defaults","d0_8":"$\\delta_0=8$\\,s","d0_16":"$\\delta_0=16$\\,s","d0_60":"$\\delta_0=60$\\,s",
     "tol_10":"Tolerance $\\pm10$\\,s","tol_20":"Tolerance $\\pm20$\\,s","tol_60":"Tolerance $\\pm60$\\,s",
     "onset_6":"Onset window 6\\,s","onset_20":"Onset window 20\\,s","sil_include":"Silencing retained","lim_modal":"Cohort modal limits"}
D={(o["channel"],o["config"],o["cell"]):o for o in out}
def cell_r(o): return "%.2f (%.2f--%.2f)"%(o["burden_ratio"],o["ci_lo"],o["ci_hi"]) if o else "--"
A=[r"\begin{table}[htbp]\centering\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",
   r"\caption{Measurement-convention scan, part A: test-set burden ratios (95\% patient-cluster bootstrap intervals, $B=1000$) under perturbations of the recorded-side merge gap, silencing handling and limit convention.}",
   r"\begin{tabular}{l"+"ll"*4+"}",r"\toprule"," & "+" & ".join(r"\multicolumn{2}{c}{%s}"%CHN[c] for c in CHN)+r" \\",
   " ".join(r"\cmidrule(lr){%d-%d}"%(2+2*j,3+2*j) for j in range(4)),
   "Convention & "+" & ".join("naive & selected" for _ in CHN)+r" \\",r"\midrule"]
for cfg in ["default","d0_8","d0_16","d0_60","sil_include","lim_modal"]:
    A.append(LAB[cfg]+" & "+" & ".join(cell_r(D.get((ch,cfg,cell))) for ch in CHN for cell in ("naive","selected"))+r" \\")
A+=[r"\bottomrule",r"\end{tabular}",
    r"\tightnote{One perturbation at a time from the fixed defaults ($\delta_0=30$\,s; silenced and inactivated samples excluded from both sides; limits carried forward from recorded changes). Matching tolerance and onset window leave burden ratios unchanged by construction and are reported in part B. For heart rate low the selected cell is the naive cell. Cohort modal limits: heart rate high 135, heart rate low 45, \SpO{} low 87, respiration rate low 8.}",
    r"\end{table}"]
B=[r"\begin{table}[htbp]\centering\scriptsize",r"\setlength{\tabcolsep}{3pt}",
   r"\caption{Measurement-convention scan, part B: episode recall and onset agreement at the selected cell under perturbations of the matching tolerance and the onset window (test set).}",
   r"\begin{tabular}{l"+"rr"*4+"}",r"\toprule"," & "+" & ".join(r"\multicolumn{2}{c}{%s}"%CHN[c] for c in CHN)+r" \\",
   " ".join(r"\cmidrule(lr){%d-%d}"%(2+2*j,3+2*j) for j in range(4)),
   "Convention & "+" & ".join("recall & onset" for _ in CHN)+r" \\",r"\midrule"]
for cfg in ["default","tol_10","tol_20","tol_60","onset_6","onset_20"]:
    cells=[]
    for ch in CHN:
        o=D.get((ch,cfg,"selected")); cells+= ["%.3f"%o["recall"],"%.3f"%o["onset_agreement"]] if o else ["--","--"]
    B.append(LAB[cfg]+" & "+" & ".join(cells)+r" \\")
B+=[r"\bottomrule",r"\end{tabular}",
    r"\tightnote{Fixed defaults: matching tolerance $\pm30$\,s, onset window 10\,s. Recall is the matched fraction of recorded runs; onset agreement is the fraction of matched pairs whose onsets (after the $\tau$ shift) fall within the window.}",
    r"\end{table}"]
open(f"{R}/sensitivity/{pre}TABLE_CONVENTION_SCAN.tex","w").write("\n".join(A)+"\n\n"+"\n".join(B)+"\n")
# the three validation criteria under each perturbation (descriptive; the held-out adjudication is not re-opened)
G={"recall":0.90,"burden":(0.80,1.25),"onset":0.80}; crit={}
for cfg in ORDER:
    passed=[]
    for ch in CHN:
        o=D.get((ch,cfg,"selected"))
        if o and o["recall"]>=G["recall"] and G["burden"][0]<=o["burden_ratio"]<=G["burden"][1] and o["onset_agreement"]>=G["onset"]: passed.append(CHN[ch])
    crit[cfg]={"channels_meeting_all_three_criteria":passed,"n":len(passed),"fewer_than_two_channels":len(passed)<2}
json.dump(crit,open(f"{R}/sensitivity/{pre}S2_CRITERIA_CHECK.json","w"),indent=1)
print("criteria under each convention:",{k:v["channels_meeting_all_three_criteria"] for k,v in crit.items()})
print("rows",len(out),"patients per cell (default naive):",{CHN[c]:D[(c,'default','naive')]['n_patients'] for c in CHN if (c,'default','naive') in D})
for ch in CHN:
    for cell in ("naive","selected"):
        o=D.get((ch,"default",cell))
        if o: print("DEFAULT CHECK",CHN[ch],cell,o["burden_ratio"],"CI",o["ci_lo"],o["ci_hi"],"recall",o["recall"],"onset",o["onset_agreement"])
for o in out:
    if o["config"]!="default": print(CHN[o["channel"]],o["cell"],o["config"],o["burden_ratio"],"x",o["ratio_to_default"])
