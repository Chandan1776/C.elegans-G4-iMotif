"""
generate_html.py  –  PGS & PIS figure from uploaded C.elegans catalogues
Aligned to match publication-style notebook regression, log-transformations, and row-wise limits.
"""
import pandas as pd, numpy as np, json, math, sys
from scipy import stats as sstats

pgs = pd.read_csv("/Users/chandanmehta/Desktop/G4_Project/RNA_Pipeline/results/PGS_Catalogue_Celegans_WBcel235.csv", low_memory=False)
pis = pd.read_csv("/Users/chandanmehta/Desktop/G4_Project/RNA_Pipeline/results/PIS_Catalogue_Celegans_WBcel235.csv", low_memory=False)

RNG = np.random.default_rng(42)
MAX_PTS = 500
MAX_SCATTER = 3000

def region_stats(df, prefix):
    regions = ["utr5","cds","utr3"]
    labels  = ["5'UTR","CDS","3'UTR"]
    density_cols = [f"{prefix}_{r}_density_per_kb" for r in regions]
    count_cols   = [f"{prefix}_{r}_count" for r in regions]

    mean_dens = [round(float(df[c].mean()), 4) for c in density_cols]
    n = len(df)
    pct_with  = [round(100.0*(df[c]>0).sum()/n, 1) for c in count_cols]

    strips = []
    for c in density_cols:
        vals = df[c][df[c]>0].values
        vals = np.log10(vals + 1)
        if len(vals) > MAX_PTS:
            vals = RNG.choice(vals, MAX_PTS, replace=False)
        strips.append(vals.tolist())

    boxes = []
    for s in strips:
        if not s:
            boxes.append({}); continue
        a  = np.array(s)
        q1, med, q3 = np.percentile(a,[25,50,75])
        iqr = q3-q1
        lo  = a[a >= q1-1.5*iqr]; hi = a[a <= q3+1.5*iqr]
        boxes.append({"q1":float(q1),"med":float(med),"q3":float(q3),
                      "wlo":float(lo.min() if len(lo) else q1),
                      "whi":float(hi.max() if len(hi) else q3)})
    return {"labels":labels,"mean_dens":mean_dens,"pct_with":pct_with,
            "strips":strips,"boxes":boxes}

def venn_counts(df, prefix):
    cnt_cols = [f"{prefix}_{r}_count" for r in ["utr5","cds","utr3"]]
    g = df.groupby("wormbase_gene_id")[cnt_cols].sum()
    a,b,c = (g.iloc[:,0]>0),(g.iloc[:,1]>0),(g.iloc[:,2]>0)
    return {
        "only_utr5":int((a&~b&~c).sum()),
        "only_cds": int((~a&b&~c).sum()),
        "only_utr3":int((~a&~b&c).sum()),
        "utr5_cds": int((a&b&~c).sum()),
        "utr5_utr3":int((a&~b&c).sum()),
        "cds_utr3": int((~a&b&c).sum()),
        "all3":     int((a&b&c).sum()),
    }

def heatmap_data(df, prefix):
    regions = ["utr5","cds","utr3"]
    labels  = ["5'UTR","CDS","3'UTR"]
    density_cols = [f"{prefix}_{r}_density_per_kb" for r in regions]
    h = df[["transcript_id"] + density_cols].copy()
    h[density_cols] = h[density_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    h["total_density"] = h[density_cols].sum(axis=1)
    h = h.sort_values("total_density", ascending=False).head(50)
    rows = []
    for _, r in h.iterrows():
        rows.append({
            "id": str(r["transcript_id"]),
            "values": [round(float(r[c]), 4) for c in density_cols],
            "total": round(float(r["total_density"]), 4),
        })
    return {"labels": labels, "rows": rows}

def scatter_data(df, x_col, dens_prefix):
    """Per-region scatter points + correlation stats matching notebook filtering 
    and log-transformations (log10(y + 0.001) for y > 0)."""
    regions = ["utr5", "cds", "utr3"]
    x_all = pd.to_numeric(df[x_col], errors="coerce")
    
    out = []
    row_all_log_y = []  # To compile all non-zero log values across regions for row-wise limits
    
    # First pass: Extract and calculate stats per region
    for r in regions:
        y_col = f"{dens_prefix}_{r}_density_per_kb"
        y_all = pd.to_numeric(df[y_col], errors="coerce")
        
        # 1. Zero-density filtering strictly applied
        mask = x_all.notna() & y_all.notna() & (y_all > 0)
        x = x_all[mask].to_numpy(dtype=float)
        y = y_all[mask].to_numpy(dtype=float)
        n = len(x)
        
        if n >= 3 and np.std(x) > 0 and np.std(y) > 0:
            # 2 & 5. Statistics and regression calculated strictly on log10(y + 0.001)
            log_y = np.log10(y + 0.001)
            row_all_log_y.extend(log_y.tolist())
            
            pear_r, pear_p = sstats.pearsonr(x, log_y)
            sp_r, sp_p = sstats.spearmanr(x, log_y)
            slope_log, intercept_log = np.polyfit(x, log_y, 1)
        else:
            pear_r = pear_p = sp_r = sp_p = slope_log = intercept_log = 0.0
            log_y = np.array([])

        idx = np.arange(n)
        if n > MAX_SCATTER:
            idx = RNG.choice(idx, MAX_SCATTER, replace=False)
        pts = [[round(float(x[i]), 3), round(float(y[i]), 3)] for i in idx]

        out.append({
            "region": r,
            "points": pts,
            "pearson_r": round(float(pear_r), 4),
            "pearson_p": float(pear_p),
            "spearman_r": round(float(sp_r), 4),
            "spearman_p": float(sp_p),
            "n": int(n),
            "slope_log": float(slope_log),
            "intercept_log": float(intercept_log),
            "xmin": float(np.min(x)) if n else 0.0,
            "xmax": float(np.max(x)) if n else 1.0,
        })
        
    # 6. Compute row-wise common Y-limits using notebook percentiles
    if row_all_log_y:
        row_y_arr = np.array(row_all_log_y)
        global_ymin_log = float(np.percentile(row_y_arr, 1) - 0.15)
        global_ymax_log = float(np.percentile(row_y_arr, 99) + 0.15)
    else:
        global_ymin_log, global_ymax_log = -3.0, 1.0
        
    for item in out:
        item["ymin_log_row"] = global_ymin_log
        item["ymax_log_row"] = global_ymax_log

    return out

pgs_s = region_stats(pgs,"pgs"); pgs_v = venn_counts(pgs,"pgs")
pis_s = region_stats(pis,"pis"); pis_v = venn_counts(pis,"pis")
pgs_h = heatmap_data(pgs,"pgs")
pis_h = heatmap_data(pis,"pis")

SCATTER_ROWS = [
    {"label": "GC% vs PIS Density", "color": "#d62728",
     "xlabel": "GC Percentage (%)", "ylabel": "PIS Density/kb",
     "data": scatter_data(pis, "gc_percentage", "pis")},
    {"label": "GC% vs PGS Density", "color": "#2e8b3d",
     "xlabel": "GC Percentage (%)", "ylabel": "PGS Density/kb",
     "data": scatter_data(pgs, "gc_percentage", "pgs")},
    {"label": "G% vs PGS Density", "color": "#1f5fd6",
     "xlabel": "G Percentage (%)", "ylabel": "PGS Density/kb",
     "data": scatter_data(pgs, "g_percentage", "pgs")},
    {"label": "C% vs PIS Density", "color": "#8a3fd6",
     "xlabel": "C Percentage (%)", "ylabel": "PIS Density/kb",
     "data": scatter_data(pis, "c_percentage", "pis")},
]

DATA = json.dumps({
    "pgs": {**pgs_s, "venn": pgs_v, "heatmap": pgs_h},
    "pis": {**pis_s, "venn": pis_v, "heatmap": pis_h},
    "scatter": SCATTER_ROWS,
})

# ── HTML ────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PGS &amp; PIS – C. elegans WBcel235</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Helvetica Neue",Arial,sans-serif;background:#fff;color:#222;padding:28px 18px 40px}
#root{max-width:1600px;margin:auto}
.master-title {
  background: white;
  color: #111;
  font-weight: 850;
  text-align: center;
  letter-spacing: .3px;
  font-size: 24px;
  padding: 0px 12px 24px;
  text-transform: uppercase;
}
.row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px;align-items:end}
.panel{display:flex;flex-direction:column;align-items:center;width:100%}
.panel-title{font-size:15px;font-weight:700;text-align:center;line-height:1.3;margin-bottom:8px;max-width:220px}
svg text{font-family:"Helvetica Neue",Arial,sans-serif}
svg{max-width:100%;height:auto}
.venn-legend{font-size:11px;border:1px solid #bbb;padding:5px 9px;margin-top:5px;
  line-height:1.75;background:#fff;font-style:italic;width:220px}
.heatmap-row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:14px;align-items:start}
.heatmap-panel{display:flex;flex-direction:column;align-items:center;min-width:0}
.heatmap-title{font-size:15px;font-weight:700;text-align:center;margin-bottom:8px}
.footer-note{font-size:12px;font-style:italic;text-align:center;color:#444;margin-top:20px}
.section-sep{height:16px}
.scatter-section{margin-top:26px;border-top:2px solid #2255bb;padding-top:14px}
.scatter-master-title{font-size:18px;font-weight:800;text-align:center;color:#1a3fa0;margin-bottom:10px}
.scatter-grid{display:grid;grid-template-columns:108px repeat(3,1fr);gap:8px 10px;align-items:center}
.scatter-col-header{font-size:14px;font-weight:700;text-align:center;color:#1a3fa0;padding-bottom:2px}
.scatter-row-label{font-size:11.5px;font-weight:700;line-height:1.35;padding-right:4px}
.scatter-cell{display:flex;justify-content:center}
</style>
</head>
<body>
<div id="root">
  <div class="master-title">cDNA Analysis of Caenorhabditis Elegans (WBcel235)</div>
</div>
<script>
const DATA=__DATA__;

const PIS_COLORS=["#e05c5c","#3cb371","#7b68ee"];
const PGS_COLORS=["#8db600","#d63384","#f4a535"];

const NS="http://www.w3.org/2000/svg";
function el(tag,a={},ch=[]){const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);for(const c of ch)e.appendChild(c);return e}
function txt(s,a={}){const t=document.createElementNS(NS,"text");for(const[k,v]of Object.entries(a))t.setAttribute(k,v);t.textContent=s;return t}
function svg(w,h){return el("svg",{width:w,height:h,viewBox:`0 0 ${w} ${h}`,xmlns:NS})}

function mulberry32(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}

function hexToRgb(hex){
  const v=hex.replace("#","");
  return [parseInt(v.slice(0,2),16),parseInt(v.slice(2,4),16),parseInt(v.slice(4,6),16)];
}
function blendHex(a,b,t){
  const x=hexToRgb(a),y=hexToRgb(b),p=Math.max(0,Math.min(1,t));
  const c=x.map((v,i)=>Math.round(v+(y[i]-v)*p));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function fmtDensity(v){
  if(v===0) return "0";
  if(v<0.01) return v.toExponential(1);
  if(v<1) return v.toFixed(2);
  return v.toFixed(1);
}

/* ── Mean Density bar ── */
function barMean(labels,values,colors){
  const W=360,H=405,ML=48,MR=8,MT=18,MB=35,cW=W-ML-MR,cH=H-MT-MB;
  const s=svg(W,H);
  const yMax=Math.max(...values)*1.15||0.001;
  s.appendChild(el("line",{x1:ML,y1:MT,x2:ML,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  s.appendChild(el("line",{x1:ML,y1:MT+cH,x2:ML+cW,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  const nTicks=8;

  for(let i=0;i<=nTicks;i++){
    const v=yMax*i/nTicks;
    const y=MT+cH-(v/yMax)*cH;
    let disp;
    if(v<0.01){
        disp=v.toFixed(3);
    }else{
        disp=v.toFixed(3).replace(/0+$/,'').replace(/\.$/,'');
    }
    s.appendChild(el("line",{x1:ML-3,y1:y,x2:ML,y2:y,stroke:"#555","stroke-width":0.7}));
    s.appendChild(txt(disp,{x:ML-5,y:y+3.5,"text-anchor":"end","font-size":"7.5",fill:"#333"}));
  }
  const yl=document.createElementNS(NS,"text");
  yl.setAttribute("transform",`translate(10,${MT+cH/2}) rotate(-90)`);
  yl.setAttribute("text-anchor","middle");yl.setAttribute("font-size","8");yl.setAttribute("fill","#333");
  yl.textContent="Mean Density (per kb)";s.appendChild(yl);
  const bw=cW/labels.length*0.52,gap=cW/labels.length;
  labels.forEach((lbl,i)=>{
    const bh=(values[i]/yMax)*cH,x=ML+i*gap+(gap-bw)/2,y=MT+cH-bh;
    s.appendChild(el("rect",{x,y,width:bw,height:bh,fill:colors[i]}));
    const disp=values[i]<0.001?values[i].toExponential(2):values[i].toFixed(3).replace(/0+$/,"");
    s.appendChild(txt(disp,{x:x+bw/2,y:y-3,"text-anchor":"middle","font-size":"8","font-weight":"600",fill:"#222"}));
    s.appendChild(txt(lbl,{x:x+bw/2,y:MT+cH+11,"text-anchor":"middle","font-size":"8.5",fill:"#333"}));
  });
  return s;
}

/* ── Strip+Box chart ── */
function stripBox(labels,strips,boxes,colors){
  const W=380,H=405,ML=48,MR=8,MT=18,MB=35,cW=W-ML-MR,cH=H-MT-MB;
  const s=svg(W,H);
  let allV=[]; strips.forEach(a=>allV=allV.concat(a));
  const yMax=allV.length?Math.max(...allV)*1.08:2;
  s.appendChild(el("line",{x1:ML,y1:MT,x2:ML,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  s.appendChild(el("line",{x1:ML,y1:MT+cH,x2:ML+cW,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  for(let i=0;i<=5;i++){
    const v=yMax*i/5,y=MT+cH-(v/yMax)*cH;
    s.appendChild(el("line",{x1:ML-3,y1:y,x2:ML,y2:y,stroke:"#555","stroke-width":0.7}));
    s.appendChild(txt(v.toFixed(1),{x:ML-5,y:y+3.5,"text-anchor":"end","font-size":"7.5",fill:"#333"}));
  }
  const yl=document.createElementNS(NS,"text");
  yl.setAttribute("transform",`translate(10,${MT+cH/2}) rotate(-90)`);
  yl.setAttribute("text-anchor","middle");yl.setAttribute("font-size","8");yl.setAttribute("fill","#333");
  yl.textContent="log₁₀(Density/kb + 1)";s.appendChild(yl);
  const colW=cW/labels.length;
  const rng=mulberry32(42);
  labels.forEach((lbl,i)=>{
    const cx=ML+i*colW+colW/2,color=colors[i],pts=strips[i],bx=boxes[i];
    if(bx&&bx.q1!==undefined){
      const bw=colW*0.36,yq1=MT+cH-(bx.q1/yMax)*cH,yq3=MT+cH-(bx.q3/yMax)*cH,
            ymed=MT+cH-(bx.med/yMax)*cH,ywlo=MT+cH-(bx.wlo/yMax)*cH,ywhi=MT+cH-(bx.whi/yMax)*cH;
      s.appendChild(el("line",{x1:cx,y1:ywhi,x2:cx,y2:yq3,stroke:color,"stroke-width":1.2}));
      s.appendChild(el("line",{x1:cx,y1:yq1,x2:cx,y2:ywlo,stroke:color,"stroke-width":1.2}));
      s.appendChild(el("line",{x1:cx-bw/2,y1:ywhi,x2:cx+bw/2,y2:ywhi,stroke:color,"stroke-width":1}));
      s.appendChild(el("line",{x1:cx-bw/2,y1:ywlo,x2:cx+bw/2,y2:ywlo,stroke:color,"stroke-width":1}));
      s.appendChild(el("rect",{x:cx-bw/2,y:yq3,width:bw,height:yq1-yq3,fill:"none",stroke:color,"stroke-width":1.4}));
    }
    pts.forEach(v=>{
      const y=MT+cH-(v/yMax)*cH, jit=(rng()-0.5)*colW*0.48;
      s.appendChild(el("circle",{cx:cx+jit,cy:y,r:1.8,fill:color,opacity:0.65}));
    });
    if(bx&&bx.q1!==undefined){
      const bw=colW*0.36,ymed=MT+cH-(bx.med/yMax)*cH;
      s.appendChild(el("line",{x1:cx-bw/2,y1:ymed,x2:cx+bw/2,y2:ymed,stroke:color,"stroke-width":2.2}));
    }
    s.appendChild(txt(lbl,{x:cx,y:MT+cH+11,"text-anchor":"middle","font-size":"8.5",fill:"#333"}));
  });
  const note=document.createElementNS(NS,"text");
  note.setAttribute("x",W/2);note.setAttribute("y",MT+cH+30);
  note.setAttribute("text-anchor","middle");note.setAttribute("font-size","7.5");
  note.setAttribute("font-style","italic");note.setAttribute("fill","#555");
  note.textContent="(0 values excluded)";s.appendChild(note);
  return s;
}

/* ── % Genes bar ── */
function barPct(labels,values,colors){
  const W=360,H=405,ML=45,MR=8,MT=18,MB=35,cW=W-ML-MR,cH=H-MT-MB;
  const s=svg(W,H);
  s.appendChild(el("line",{x1:ML,y1:MT,x2:ML,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  s.appendChild(el("line",{x1:ML,y1:MT+cH,x2:ML+cW,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  const yMax=Math.max(...values)*1.15;
  const nTicks=5;
  for(let i=0;i<=nTicks;i++){
    const v=yMax*i/nTicks;
    const y=MT+cH-(v/yMax)*cH;
    s.appendChild(el("line",{x1:ML-3,y1:y,x2:ML,y2:y,stroke:"#555","stroke-width":0.7}));
    s.appendChild(txt(v.toFixed(1),{x:ML-5,y:y+3.5,"text-anchor":"end","font-size":"8",fill:"#333"}));
  }
  const yl=document.createElementNS(NS,"text");
  yl.setAttribute("transform",`translate(10,${MT+cH/2}) rotate(-90)`);
  yl.setAttribute("text-anchor","middle");yl.setAttribute("font-size","8");yl.setAttribute("fill","#333");
  yl.textContent="% Genes";s.appendChild(yl);
  const bw=cW/labels.length*0.52,gap=cW/labels.length;
  labels.forEach((lbl,i)=>{
    const bh=(values[i]/yMax)*cH,x=ML+i*gap+(gap-bw)/2,y=MT+cH-bh;
    s.appendChild(el("rect",{x,y,width:bw,height:bh,fill:colors[i]}));
    s.appendChild(txt(values[i].toFixed(1)+"%",{x:x+bw/2,y:y-3,"text-anchor":"middle","font-size":"8","font-weight":"600",fill:"#222"}));
    s.appendChild(txt(lbl,{x:x+bw/2,y:MT+cH+11,"text-anchor":"middle","font-size":"8.5",fill:"#333"}));
  });
  return s;
}

/* ── Venn ── */
function vennSvg(v,c0,c1,c2){
  const W=320,H=255;
  const s=svg(W,H);
  const r=78,cx=W/2,cy=110,dx=42,dy=28;
  const p5=[cx-dx,cy-dy],pC=[cx+dx,cy-dy],p3=[cx,cy+dy+12];
  const A="99";
  s.appendChild(el("circle",{cx:p5[0],cy:p5[1],r,fill:c0+A,stroke:c0,"stroke-width":"2"}));
  s.appendChild(el("circle",{cx:pC[0],cy:pC[1],r,fill:c1+A,stroke:c1,"stroke-width":"2"}));
  s.appendChild(el("circle",{cx:p3[0],cy:p3[1],r,fill:c2+A,stroke:c2,"stroke-width":"2"}));
  const LS={"font-size":"13","font-weight":"700","text-anchor":"middle"};
  s.appendChild(txt("5'UTR",{...LS,x:p5[0]-36,y:p5[1]-44,fill:c0}));
  s.appendChild(txt("CDS",  {...LS,x:pC[0]+36,y:pC[1]-44,fill:c1}));
  s.appendChild(txt("3'UTR",{...LS,x:p3[0],  y:p3[1]+52, fill:c2}));
  const NS2={"font-size":"12","font-weight":"600","text-anchor":"middle",fill:"#111"};
  s.appendChild(txt(v.only_utr5+"",{...NS2,x:p5[0]-30,y:p5[1]+3}));
  s.appendChild(txt(v.only_cds+"", {...NS2,x:pC[0]+30,y:pC[1]+3}));
  s.appendChild(txt(v.only_utr3+"",{...NS2,x:p3[0],   y:p3[1]+28}));
  s.appendChild(txt(v.utr5_cds+"", {...NS2,x:(p5[0]+pC[0])/2,y:p5[1]-10}));
  s.appendChild(txt(v.utr5_utr3+"",{...NS2,x:(p5[0]+p3[0])/2-10,y:(p5[1]+p3[1])/2+15}));
  s.appendChild(txt(v.cds_utr3+"", {...NS2,x:(pC[0]+p3[0])/2+10,y:(pC[1]+p3[1])/2+15}));
  s.appendChild(txt(v.all3+"",     {...NS2,x:cx,y:cy+5}));
  return s;
}

function legend(v){
  const d=document.createElement("div");
  d.className="venn-legend";
  d.innerHTML=`Only 5'UTR: ${v.only_utr5}<br>Only CDS: ${v.only_cds}<br>Only 3'UTR: ${v.only_utr3}<br>5'UTR ∩ CDS only: ${v.utr5_cds}<br>5'UTR ∩ 3'UTR only: ${v.utr5_utr3}<br>CDS ∩ 3'UTR only: ${v.cds_utr3}<br>All three (5'UTR ∩ CDS ∩ 3'UTR): ${v.all3}`;
  return d;
}

function heatmapSvg(hm,colors){
  const rows=hm.rows,W=760,rowH=10,ML=155,MR=24,MT=26,headerH=36,MB=46;
  const cellW=(W-ML-MR)/hm.labels.length;
  const H=MT+headerH+rows.length*rowH+MB;
  const s=svg(W,H);
  const vals=rows.flatMap(r=>r.values.map(v=>Math.log10(v+1)));
  const maxLog=Math.max(...vals,0.001);

  hm.labels.forEach((lbl,i)=>{
    const x=ML+i*cellW+cellW/2;
    s.appendChild(txt(lbl,{x,y:MT+12,"text-anchor":"middle","font-size":"10","font-weight":"700",fill:"#333"}));
  });
  s.appendChild(txt("Transcript ID",{x:ML-8,y:MT+12,"text-anchor":"end","font-size":"10","font-weight":"700",fill:"#333"}));
  s.appendChild(txt("Density per kb (color scaled as log₁₀(value + 1))",{x:ML,y:H-14,"font-size":"9","font-style":"italic",fill:"#555"}));

  rows.forEach((r,ri)=>{
    const y=MT+headerH+ri*rowH;
    s.appendChild(txt(r.id,{x:ML-8,y:y+rowH-2,"text-anchor":"end","font-size":"7.6",fill:"#333"}));
    r.values.forEach((v,i)=>{
      const t=Math.log10(v+1)/maxLog;
      const fill=v>0?blendHex("#ffffff",colors[i],0.12+0.88*t):"#f4f4f4";
      const cx=ML+i*cellW;
      const rect=el("rect",{x:cx+1,y,width:cellW-2,height:rowH-1,fill,stroke:"#ffffff","stroke-width":0.5});
      const title=document.createElementNS(NS,"title");
      title.textContent=`${r.id} ${hm.labels[i]}: ${fmtDensity(v)} per kb`;
      rect.appendChild(title);
      s.appendChild(rect);
      const textColor=t>0.55?"#ffffff":"#222222";
      s.appendChild(txt(fmtDensity(v),{x:cx+cellW/2,y:y+rowH-2,"text-anchor":"middle","font-size":"6.3",fill:textColor}));
    });
  });

  const gx=W-196,gy=H-21,gw=140,gh=8;
  for(let i=0;i<28;i++){
    s.appendChild(el("rect",{x:gx+i*gw/28,y:gy,width:gw/28+0.5,height:gh,fill:blendHex("#ffffff","#444444",i/27)}));
  }
  s.appendChild(el("rect",{x:gx,y:gy,width:gw,height:gh,fill:"none",stroke:"#777","stroke-width":0.6}));
  s.appendChild(txt("low",{x:gx-4,y:gy+8,"text-anchor":"end","font-size":"8",fill:"#555"}));
  s.appendChild(txt("high",{x:gx+gw+4,y:gy+8,"font-size":"8",fill:"#555"}));
  return s;
}

function fmtP(p){
  if(p<0.001) return "P < 0.001";
  return "P = "+p.toFixed(3);
}
function fmtN(n){ return n.toLocaleString("en-US"); }

/* ── 3, 4, 6 & 7. Refactored Publication Scatter Panel ── */
function scatterSvg(d,color,xlabel,ylabel){
  const W=300,H=222,ML=40,MR=12,MT=10,MB=32,cW=W-ML-MR,cH=H-MT-MB;
  const s=svg(W,H);
  
  const xMin=Math.min(0,d.xmin), xMax=(d.xmax||1)*1.05;
  // 6. Implements Row-wise Global Limits passed from backend
  const yMinLog = d.ymin_log_row;
  const yMaxLog = d.ymax_log_row;
  const logRange = yMaxLog - yMinLog;

  // 7. Light gray coordinate grids
  const xTicks=5;
  for(let i=0;i<=xTicks;i++){
    const v=xMin+(xMax-xMin)*i/xTicks;
    const x=ML+(v-xMin)/(xMax-xMin)*cW;
    s.appendChild(el("line",{x1:x,y1:MT,x2:x,y2:MT+cH,stroke:"#e8e8e8","stroke-width":0.5}));
    s.appendChild(el("line",{x1:x,y1:MT+cH,x2:x,y2:MT+cH+3,stroke:"#555","stroke-width":0.7}));
    s.appendChild(txt(v.toFixed(0),{x,y:MT+cH+12,"text-anchor":"middle","font-size":"7.5",fill:"#333"}));
  }
  
  const yTicks=4;
  for(let i=0;i<=yTicks;i++){
    const v=yMinLog+(logRange)*i/yTicks;
    const y=MT+cH-((v-yMinLog)/logRange)*cH;
    s.appendChild(el("line",{x1:ML,y1:y,x2:ML+cW,y2:y,stroke:"#e8e8e8","stroke-width":0.5}));
    s.appendChild(el("line",{x1:ML-3,y1:y,x2:ML,y2:y,stroke:"#555","stroke-width":0.7}));
    s.appendChild(txt(v.toFixed(1),{x:ML-5,y:y+2.6,"text-anchor":"end","font-size":"7.5",fill:"#333"}));
  }

  // 7. Hide top/right boundaries (clean scientific axis framework)
  s.appendChild(el("line",{x1:ML,y1:MT,x2:ML,y2:MT+cH,stroke:"#555","stroke-width":0.8}));
  s.appendChild(el("line",{x1:ML,y1:MT+cH,x2:ML+cW,y2:MT+cH,stroke:"#555","stroke-width":0.8}));

  s.appendChild(txt(xlabel,{x:ML+cW/2,y:H-2,"text-anchor":"middle","font-size":"8",fill:"#222"}));
  
  // 4. Clean baseline metric naming shift
  const yl=document.createElementNS(NS,"text");
  yl.setAttribute("transform",`translate(10,${MT+cH/2}) rotate(-90)`);
  yl.setAttribute("text-anchor","middle");yl.setAttribute("font-size","7.5");yl.setAttribute("fill","rgb(34,34,34)");
  yl.textContent=`log\u2081\u2080(${ylabel} + 0.001)`; s.appendChild(yl);

  // 3 & 7. Render point coordinates shifted to notebook specs
  d.points.forEach(p=>{
    const x=ML+(p[0]-xMin)/(xMax-xMin)*cW;
    const yLogV=Math.log10(p[1]+0.001);
    const clampedYLog = Math.max(yMinLog, Math.min(yMaxLog, yLogV));
    const y=MT+cH-((clampedYLog-yMinLog)/logRange)*cH;
    s.appendChild(el("circle",{cx:x,cy:y,r:0.8,fill:color,opacity:0.42}));
  });

  // 2 & 7. Solid gray publication regression line
  const yAt0=d.intercept_log+d.slope_log*xMin, yAtMax=d.intercept_log+d.slope_log*xMax;
  const ly0=MT+cH-((Math.max(yMinLog,Math.min(yMaxLog,yAt0))-yMinLog)/logRange)*cH;
  const ly1=MT+cH-((Math.max(yMinLog,Math.min(yMaxLog,yAtMax))-yMinLog)/logRange)*cH;
  s.appendChild(el("line",{x1:ML,y1:ly0,x2:ML+cW,y2:ly1,stroke:"#555555","stroke-width":1.5}));

  // Overlay stats metrics panel transparently
  const bw=92,bh=42,bx=ML+6,by=MT+3;
  s.appendChild(el("rect",{x:bx,y:by,width:bw,height:bh,fill:"#ffffff",stroke:"#bbb","stroke-width":0.6,opacity:0.9}));
  const lines=[
    `Pearson r = ${d.pearson_r.toFixed(2)}`,
    `Spearman \u03c1 = ${d.spearman_r.toFixed(2)}`,
    fmtP(d.pearson_p),
    `n = ${fmtN(d.n)}`
  ];
  lines.forEach((t,i)=>{
    s.appendChild(txt(t,{x:bx+4,y:by+10+i*9.5,"font-size":"6.8",fill:"#222"}));
  });
  return s;
}

function buildScatterSection(){
  const wrap=document.createElement("div"); wrap.className="scatter-section";
  const title=document.createElement("div"); title.className="scatter-master-title";
  title.textContent="Sequence Composition vs Motif Density (motifs per kb)";
  wrap.appendChild(title);

  const grid=document.createElement("div"); grid.className="scatter-grid";
  grid.appendChild(document.createElement("div"));
  ["5'UTR","CDS","3'UTR"].forEach(lbl=>{
    const h=document.createElement("div"); h.className="scatter-col-header"; h.textContent=lbl;
    grid.appendChild(h);
  });
  DATA.scatter.forEach(row=>{
    const lab=document.createElement("div"); lab.className="scatter-row-label";
    lab.style.color=row.color;
    lab.innerHTML=row.label.replace(" vs ","<br>")===row.label?row.label:row.label.split(" vs ").join("<br>vs ");
    grid.appendChild(lab);
    row.data.forEach(d=>{
      const cell=document.createElement("div"); cell.className="scatter-cell";
      cell.appendChild(scatterSvg(d,row.color,row.xlabel,row.ylabel));
      grid.appendChild(cell);
    });
  });
  wrap.appendChild(grid);
  return wrap;
}
function panel(title,el){
  const d=document.createElement("div"); d.className="panel";
  const h=document.createElement("div"); h.className="panel-title"; h.innerHTML=title;
  d.appendChild(h); d.appendChild(el); return d;
}

function buildRow(d,colors,tPrefix){
  const row=document.createElement("div"); row.className="row";
  row.appendChild(panel(`${tPrefix} Mean Density`,     barMean(d.labels,d.mean_dens,colors)));
  row.appendChild(panel(`${tPrefix} Density Distribution`, stripBox(d.labels,d.strips,d.boxes,colors)));
  row.appendChild(panel(`${tPrefix} % Genes with Motif`,        barPct(d.labels,d.pct_with,colors)));
  const vp=document.createElement("div"); vp.className="panel";
  const vh=document.createElement("div"); vh.className="panel-title";
  vh.innerHTML=`${tPrefix} Region Overlap`;
  vp.appendChild(vh);
  vp.appendChild(vennSvg(d.venn,colors[0],colors[1],colors[2]));
  vp.appendChild(legend(d.venn));
  row.appendChild(vp);
  return row;
}

function heatmapPanel(title,hm,colors){
  const d=document.createElement("div"); d.className="heatmap-panel";
  const h=document.createElement("div"); h.className="heatmap-title"; h.textContent=title;
  d.appendChild(h); d.appendChild(heatmapSvg(hm,colors));
  return d;
}

function buildHeatmaps(){
  const row=document.createElement("div"); row.className="heatmap-row";
  row.appendChild(heatmapPanel("Top 50 PIS-rich transcripts",DATA.pis.heatmap,PIS_COLORS));
  row.appendChild(heatmapPanel("Top 50 PGS-rich transcripts",DATA.pgs.heatmap,PGS_COLORS));
  return row;
}

const root=document.getElementById("root");
root.appendChild(buildRow(DATA.pis,PIS_COLORS,"PIS"));
const sep=document.createElement("div"); sep.className="section-sep"; root.appendChild(sep);
root.appendChild(buildRow(DATA.pgs,PGS_COLORS,"PGS"));
root.appendChild(buildHeatmaps());
root.appendChild(buildScatterSection());
const fn=document.createElement("p"); fn.className="footer-note";
fn.textContent="Note: For density distribution plots, transcripts with zero motif density were excluded before log transformation. Heatmaps rank transcripts by total density across 5'UTR, CDS, and 3'UTR. Scatter plots show per-transcript composition vs. motif density with a fitted linear regression line; correlation statistics (Pearson r, Spearman \u03c1, p-value, n) are computed on the full non-zero dataset using log\u2081\u2080(density + 0.001), while plotted points are subsampled for rendering. Subplots share a row-wise common Y-axis dynamically scaled to publication layout targets.";
root.appendChild(fn);
</script>
</body>
</html>"""

html_out = HTML.replace("__DATA__", DATA)
with open("/Users/chandanmehta/Desktop/G4_Project/RNA_Pipeline/scripts/pgs_pis_figure.html","w") as f:
    f.write(html_out)
print("Done. PGS venn:", pgs_v)
print("PIS venn:", pis_v)