#!/usr/bin/env python3
"""Recorded burden and silencing tables from the alarm-epidemiology scan rows.
"""
import csv,glob,os,collections,json
from alarmreplay.paths import WORK as R, COHORT as SAMP
splitmap={r["aMRN"]:r["split"] for r in csv.DictReader(open(f"{R}/design/SPLIT_ASSIGNMENT.csv"))}
tiers={}
for r in csv.DictReader(open(SAMP)):
    tiers[r["aCSN"]]=r["tier"]
seen=set()
K=collections.defaultdict(lambda: collections.Counter())      # (klass) totals
CHN=collections.defaultdict(lambda: collections.Counter())    # per base alarm
ADM=collections.Counter(); DAYS=collections.Counter()
TIER=collections.defaultdict(lambda: collections.Counter())
n_adm=0; n_noalarm=0; n_err=0
for fp in sorted(glob.glob(f"{R}/alarm_epidemiology/a1_rows_*.csv")):
    for r in csv.DictReader(open(fp)):
        key=(r["aMRN"],r["aCSN"],r["alarm_base"])
        if key in seen: continue
        seen.add(key)
        if r["alarm_base"]=="__NO_ALARM_FILE__": n_noalarm+=1; continue
        if r["alarm_base"]=="__READ_ERROR__": n_err+=1; continue
        akey=(r["aMRN"],r["aCSN"])
        if akey not in ADM:
            ADM[akey]=1; n_adm+=1
            span=(int(r["adm_last_s"])-int(r["adm_first_s"]))/86400 if r["adm_first_s"] else 0
            DAYS[akey]=max(span,1/24)
        kl=r["klass"]; d=K[kl]
        rows=int(r["n_rows"]); runs=int(r["n_runs"]); rsec=int(r["run_seconds"])
        sil=int(r["n_sil"]); ina=int(r["n_inact"])
        for tgt in (d, CHN[r["alarm_base"]], TIER[tiers.get(r["aCSN"],"?")]):
            tgt["rows"]+=rows; tgt["runs"]+=runs; tgt["rsec"]+=rsec; tgt["sil"]+=sil; tgt["ina"]+=ina
        d["adm"]+=0
total_days=sum(DAYS.values())
out={"admissions_scanned":n_adm,"no_alarm_file":n_noalarm,"read_errors":n_err,
     "total_admission_days_spanned":round(total_days,1),"by_class":{},"top_alarms":{},"by_tier":{}}
for kl,d in sorted(K.items()):
    out["by_class"][kl]={"annunciation_runs":d["runs"],"runs_per_adm_day":round(d["runs"]/total_days,2),
      "alarm_seconds":d["rsec"],"alarm_min_per_adm_day":round(d["rsec"]/60/total_days,2),
      "silenced_row_share":round(d["sil"]/max(d["rows"],1),4),
      "silenced_share_of_alarm_time":round(2*d["sil"]/max(d["rsec"],1),4),
      "inactivated_row_share":round(d["ina"]/max(d["rows"],1),4)}
for base,d in sorted(CHN.items(),key=lambda kv:-kv[1]["runs"])[:20]:
    out["top_alarms"][base]={"runs":d["runs"],"runs_per_adm_day":round(d["runs"]/total_days,3),
      "silenced_share_of_alarm_time":round(2*d["sil"]/max(d["rsec"],1),4)}
for t,d in sorted(TIER.items()):
    out["by_tier"][t]={"runs":d["runs"],"alarm_seconds":d["rsec"],"silenced_share":round(2*d["sil"]/max(d["rsec"],1),4)}
json.dump(out,open(f"{R}/alarm_epidemiology/A1_SUMMARY.json","w"),indent=1)
rows=[["class","runs","runs_per_adm_day","alarm_min_per_adm_day","silenced_share_of_alarm_time","inactivated_row_share"]]
for kl,d in out["by_class"].items():
    rows.append([kl,d["annunciation_runs"],d["runs_per_adm_day"],d["alarm_min_per_adm_day"],
                 d["silenced_share_of_alarm_time"],d["inactivated_row_share"]])
with open(f"{R}/alarm_epidemiology/A1_BURDEN_BY_CLASS.csv","w",newline="") as f: csv.writer(f).writerows(rows)
print(json.dumps({k:v for k,v in out.items() if k!="top_alarms"},indent=1)[:1200])
print("top alarms:",list(out["top_alarms"])[:8])
