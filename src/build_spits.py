import json, pandas as pd
from sklearn.model_selection import train_test_split

with open("audioset_index.json") as f:
    IDX = json.load(f)
print(f"{len(IDX):,} clips local")

TRIGGER_MAP = {
    "/m/03qc9zr":"Scream","/m/07p6fty":"Shout","/t/dd00135":"Shout",
    "/m/0463cq4":"Crying","/t/dd00002":"Crying","/m/014zdl":"Explosion",
    "/m/0g6b5":"Explosion","/m/032s66":"Gunshot","/m/039jq":"Glass",
    "/m/03kmc9":"Siren","/m/04qvtq":"Siren","/m/012n7d":"Siren",
    "/m/012ndj":"Siren","/m/07pp_mv":"Alarm","/m/02mfyn":"Alarm",
    "/m/01y3hg":"Alarm","/m/0c3f7m":"Alarm",
}
CLASSES=["Scream","Shout","Crying","Explosion","Gunshot","Glass","Siren","Alarm"]
tids=set(TRIGGER_MAP.keys())

df=pd.read_csv("unbalanced_train_segments.csv",skiprows=3,header=None,
    names=["ytid","start_s","end_s","labels"],quotechar='"',
    skipinitialspace=True,comment="#")
df["ytid"]=df["ytid"].str.strip()
df=df[df["ytid"].isin(IDX)]
print(f"{len(df):,} local w/ metadata")

def has_t(s): return bool(set(s.strip('"').split(","))&tids)
dt=df[df["labels"].apply(has_t)].copy()
for c in CLASSES:
    cids=[a for a,v in TRIGGER_MAP.items() if v==c]
    dt[c]=dt["labels"].apply(lambda x:1 if any(a in x for a in cids) else 0)

print(dt[CLASSES].sum().sort_values(ascending=False))

TARGET=300
smp=[]
for c in CLASSES:
    cd=dt[dt[c]==1]
    n=min(TARGET,len(cd))
    smp.append(cd.sample(n=n,random_state=42))
    print(f"  {c}: {n}")
dp=pd.concat(smp).drop_duplicates(subset="ytid")

HN={"/m/09x0r","/m/04rlf","/m/07qmpdm","/m/03qtwd","/m/0261r1"}
def cn(s): return len(set(s.strip('"').split(","))&tids)==0
def hn(s): return bool(set(s.strip('"').split(","))&HN)
dn=df[~df["ytid"].isin(set(dp["ytid"]))&df["labels"].apply(cn)]
dh=dn[dn["labels"].apply(hn)].sample(n=600,random_state=42).copy()
dh[CLASSES]=0

da=pd.concat([dp,dh],ignore_index=True).sample(frac=1,random_state=42)
tr,tmp=train_test_split(da,test_size=0.30,random_state=42)
va,te=train_test_split(tmp,test_size=0.50,random_state=42)
for nm,d in [("train",tr),("val",va),("test",te)]:
    d.to_csv(f"audioset_{nm}.csv",index=False)
    print(f"  audioset_{nm}.csv: {len(d)}")
print("Done.")
