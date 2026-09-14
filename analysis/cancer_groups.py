#!/usr/bin/env python3
"""Coarse cancer group per admission from the release's diagnosis fields: haematological,
thoracic, gastrointestinal, other solid. Descriptive only.
"""
import csv,gzip,glob,os,collections,re
from alarmreplay.paths import WORK as R, COHORT as SAMP, DIAGNOSES as S0, CANCER_SITES
COLS=["aCSN","aMRN","admsnOffset","dischOffset","Admission_Type","Admit_Source","LOS_Hours","REASON_VISIT_NAME","Discharge_Destination","Primary_ICD10","Primary_Diagnosis","Secondary_ICD10","Secondary_Diagnosis"]
iP,iS=COLS.index("Primary_ICD10"),COLS.index("Secondary_ICD10")
tele={r["aCSN"] for r in csv.DictReader(open(SAMP))}
pat=re.compile(r"^C(\d{2}|4A|7A|7B)")
def group(code):
    m=pat.match(code.strip().upper())
    if not m: return None
    k=m.group(1)
    if k in ("4A","7A","7B"): return "other_solid"
    n=int(k)
    if 81<=n<=96: return "haematological"
    if 30<=n<=39: return "thoracic"
    if 15<=n<=26: return "gastrointestinal"
    return "other_solid"
out={}; first={}
for fp in sorted(glob.glob(f"{S0}/part_*.csv.gz")):
    with gzip.open(fp,"rt",newline="") as f:
        for row in csv.reader(f):
            if len(row)<len(COLS) or row[0] not in tele: continue
            g=None; fc=None
            for field in (row[iP],row[iS]):
                for code in field.split(","):
                    gg=group(code)
                    if gg: g=gg; fc=code.strip(); break
                if g: break
            if g and row[0] not in out: out[row[0]]=g; first[row[0]]=fc
os.makedirs(f"{R}/cache",exist_ok=True)
with open(f"{R}/cache/cancer_group_by_admission.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["aCSN","cancer_group","first_malignant_code"])
    for a in sorted(out,key=int): w.writerow([a,out[a],first[a]])
print("malignant-code admissions in telemetry cohort:",len(out),collections.Counter(out.values()))
# agreement with the earlier cancer-site labels
p2={r["aCSN"]:r["cancer_site"] for r in csv.DictReader(open(CANCER_SITES)) if r["cancer_site"]} if os.path.exists(CANCER_SITES) else {}
mp={"Heme":"haematological","Thoracic":"thoracic","GI":"gastrointestinal","OtherSolid":"other_solid"}
agree=collections.Counter()
for a,s in p2.items(): agree[(mp[s],out.get(a,"NONE"))]+=1
ok=sum(v for (x,y),v in agree.items() if x==y); tot=sum(agree.values())
print("agreement with the earlier cancer-site labels:",ok,"/",tot,round(ok/tot,3)); print({k:v for k,v in agree.items() if k[0]!=k[1]})
