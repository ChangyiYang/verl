import re, sys
KEYS=["critic/rewards/mean","critic/score/mean","actor/pg_loss","actor/ppo_kl","actor/grad_norm","response_length/mean","actor/lr"]
def parse(path):
    rows={}
    for line in open(path, errors="ignore"):
        line=re.sub(r'\x1b\[[0-9;]*m','',line)
        m=re.search(r'step:(\d+) - ',line)
        if not m: continue
        s=int(m.group(1)); d={}
        for k in KEYS:
            mm=re.search(re.escape(k)+r':(?:np\.float64\()?(-?[0-9.eE+]+)\)?',line)
            if mm:
                try: d[k]=float(mm.group(1))
                except: pass
        if d: rows[s]=d
    return rows
if len(sys.argv)==2:
    r=parse(sys.argv[1]); print(f"{sys.argv[1]}: {len(r)} steps")
    for s in sorted(r):
        if s in (1,5,10,20,30,40,50):
            d=r[s]; print("step %2d: reward=%.4f pg_loss=%+.4e grad_norm=%.4e resp_len=%.1f"%(s,d.get('critic/rewards/mean',-1),d.get('actor/pg_loss',0),d.get('actor/grad_norm',0),d.get('response_length/mean',-1)))
else:
    a=parse(sys.argv[1]); b=parse(sys.argv[2])
    print("step | full(reward,pg_loss,gnorm,rlen) | delta(...) | reward_match")
    for s in sorted(set(a)&set(b)):
        if s%10 and s not in(1,5): continue
        fa,fb=a[s],b[s]
        print("%2d | %.4f %+.3e %.3e %.0f | %.4f %+.3e %.3e %.0f"%(s,
          fa.get('critic/rewards/mean',-1),fa.get('actor/pg_loss',0),fa.get('actor/grad_norm',0),fa.get('response_length/mean',-1),
          fb.get('critic/rewards/mean',-1),fb.get('actor/pg_loss',0),fb.get('actor/grad_norm',0),fb.get('response_length/mean',-1)))
