"""
Generate an editable, self-contained HTML genomic motif atlas panel.

The HTML uses inline SVG elements only. No Plotly, no external JavaScript,
and no imported plot images.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_DIR = PROJECT_ROOT / "results"
if not BASE_DIR.exists():
    BASE_DIR = Path("/Users/chandanmehta/Desktop/G4_Project/DNA_Pipeline/results")
PIS_FILE = BASE_DIR / "PIS_Catalogue_DNA_Celegans_WBcel235.csv"
PGS_FILE = BASE_DIR / "PGS_Catalogue_DNA_Celegans_WBcel235.csv"
SCAN_STATS_FILE = BASE_DIR / "scan_stats_DNA_Celegans_WBcel235.json"

OUT_DIR = BASE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIONS = [
    ("Promoter", "promoter"),
    ("5'UTR", "five_prime_UTR"),
    ("CDS", "CDS"),
    ("Intron", "intron"),
    ("3'UTR", "three_prime_UTR"),
    ("Intergenic", "intergenic"),
]

EXONIC = [("5'UTR", "five_prime_UTR"), ("CDS", "CDS"), ("3'UTR", "three_prime_UTR")]
GENE_BODY = [("5'UTR", "five_prime_UTR"), ("CDS", "CDS"), ("Intron", "intron"), ("3'UTR", "three_prime_UTR")]
HEATMAP_REGIONS = [
    ("Promoter", "promoter"),
    ("5'UTR", "five_prime_UTR"),
    ("CDS", "CDS"),
    ("Intron", "intron"),
    ("3'UTR", "three_prime_UTR"),
    ("Gene Body", "gene_body"),
    ("Intergenic", "intergenic"),
]

CHR_ORDER = ["I", "II", "III", "IV", "V", "X"]
CHR_LABELS = {"I": "I", "II": "II", "III": "III", "IV": "IV", "V": "V", "X": "X"}

# Lighter manuscript palette:
PIS = ["#ff6b5f", "#26b99a", "#8d5cf6", "#ffad4d", "#f06aa6", "#c7b8ff"]
PGS = ["#9bdc13", "#f35ca8", "#ffad20", "#65c56f", "#7ccfc0", "#ffd166"]
BLUE = "#082b6f"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in df.columns:
        if c.endswith("_count") or c.endswith("_density_per_kb") or c == "gene_length":
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def load_stats() -> dict:
    if SCAN_STATS_FILE.exists():
        return json.loads(SCAN_STATS_FILE.read_text())
    return {}


def ccol(prefix: str, region: str) -> str:
    return f"{prefix}_{region}_count"


def dcol(prefix: str, region: str) -> str:
    return f"{prefix}_{region}_density_per_kb"


def density_values(df: pd.DataFrame, prefix: str, region: str, stats: dict) -> tuple[np.ndarray, bool]:
    col = dcol(prefix, region)
    if col in df.columns:
        return df[col].to_numpy(float), False

    count = ccol(prefix, region)
    if count not in df.columns:
        return np.zeros(len(df)), True

    if region == "intergenic":
        genome_bp = sum(v.get("length_bp", 0) for k, v in stats.get("per_chromosome", {}).items() if k in CHR_ORDER)
        val = df[count].sum() / genome_bp * 1000 if genome_bp else 0
        return np.array([val], dtype=float), True

    denom = df["gene_length"].replace(0, np.nan)
    vals = (df[count] / denom * 1000).replace([np.inf, -np.inf], np.nan).fillna(0)
    return vals.to_numpy(float), True


def mean_density(df: pd.DataFrame, prefix: str, stats: dict) -> tuple[list[float], list[bool]]:
    vals, fallback = [], []
    for _, region in REGIONS:
        v, fb = density_values(df, prefix, region, stats)
        vals.append(float(np.nanmean(v)) if len(v) else 0.0)
        fallback.append(fb)
    return vals, fallback


def dist_values(df: pd.DataFrame, prefix: str, stats: dict) -> list[np.ndarray]:
    out = []
    for _, region in REGIONS:
        v, _ = density_values(df, prefix, region, stats)
        v = v[np.isfinite(v) & (v > 0)]
        out.append(np.log10(v + 1))
    return out


def pct_values(df: pd.DataFrame, prefix: str) -> list[float]:
    n = len(df)
    vals = []
    for _, region in REGIONS:
        col = ccol(prefix, region)
        vals.append(100 * float((df[col] > 0).sum()) / n if col in df.columns and n else 0)
    return vals


def region_sets(df: pd.DataFrame, prefix: str) -> dict[str, set]:
    sets = {}
    for label, region in REGIONS:
        col = ccol(prefix, region)
        sets[label] = set(df.loc[df[col].fillna(0) > 0, "wormbase_gene_id"]) if col in df.columns else set()
    return sets


def overlap3(df: pd.DataFrame, prefix: str) -> dict[str, int]:
    prom = set(df.loc[df[ccol(prefix, "promoter")].fillna(0) > 0, "wormbase_gene_id"])
    body_counts = sum(df[ccol(prefix, r)].fillna(0) for _, r in GENE_BODY)
    body = set(df.loc[body_counts > 0, "wormbase_gene_id"])
    inter = set(df.loc[df[ccol(prefix, "intergenic")].fillna(0) > 0, "wormbase_gene_id"])
    return {
        "only_promoter": len(prom - body - inter),
        "only_body": len(body - prom - inter),
        "only_intergenic": len(inter - prom - body),
        "promoter_body": len((prom & body) - inter),
        "promoter_intergenic": len((prom & inter) - body),
        "body_intergenic": len((body & inter) - prom),
        "all_three": len(prom & body & inter),
        "promoter_total": len(prom),
        "body_total": len(body),
        "intergenic_total": len(inter),
    }


def sum_counts(df: pd.DataFrame, prefix: str, regions: list[tuple[str, str]]) -> list[int]:
    return [int(df[ccol(prefix, r)].fillna(0).sum()) for _, r in regions]


def chr_density(stats: dict) -> tuple[list[str], list[float], list[float]]:
    labels, pis, pgs = [], [], []
    for chrom in CHR_ORDER:
        d = stats.get("per_chromosome", {}).get(chrom)
        if not d:
            continue
        length = d.get("length_bp", 0)
        if length <= 0:
            continue
        labels.append(CHR_LABELS.get(chrom, chrom))
        pis.append(d.get("pis_hits", 0) / length * 1000)
        pgs.append(d.get("pgs_hits", 0) / length * 1000)
    return labels, pis, pgs


def density_series_for_heatmap(df: pd.DataFrame, prefix: str, region: str) -> pd.Series:
    col = dcol(prefix, region)
    if col in df.columns:
        return df[col].fillna(0)

    count = ccol(prefix, region)
    if count in df.columns:
        denom = df["gene_length"].replace(0, np.nan)
        return (df[count] / denom * 1000).replace([np.inf, -np.inf], np.nan).fillna(0)

    return pd.Series(np.zeros(len(df)), index=df.index)


def top50_heatmap_matrix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    top = df.sort_values(f"{prefix}_total_density_per_kb", ascending=False).head(50).copy()
    mat = pd.DataFrame(
        {
            label: np.log10(density_series_for_heatmap(top, prefix, region).to_numpy(float) + 1)
            for label, region in HEATMAP_REGIONS
        },
        index=top["wormbase_gene_id"].astype(str),
    )
    return mat


def cluster_rows(mat: pd.DataFrame) -> pd.DataFrame:
    if len(mat) <= 2:
        return mat
    distances = pdist(mat.to_numpy(), metric="euclidean")
    if np.allclose(distances, 0):
        return mat
    order = leaves_list(linkage(distances, method="average"))
    return mat.iloc[order]


def light_heatmap_hex(value: float, vmin: float, vmax: float, mode: str = "pis") -> str:
    t = 0 if vmax <= vmin else min(max((value - vmin) / (vmax - vmin), 0), 1)
    
    if mode == "pis":
        stops = [
            (0.00, (250, 250, 250)),
            (0.25, (254, 235, 232)),
            (0.65, (252, 171, 157)),
            (1.00, (235, 75, 75))
        ]
    else:
        stops = [
            (0.00, (250, 250, 250)),
            (0.25, (232, 247, 235)),
            (0.65, (155, 222, 163)),
            (1.00, (65, 170, 80))
        ]
        
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            rgb = tuple(round(c0[i] + f * (c1[i] - c0[i])) for i in range(3))
            return "#%02x%02x%02x" % rgb
    return "#41aa50" if mode == "pgs" else "#eb4b4b"


def svg_text(x, y, text, cls="", **attrs) -> str:
    extra = " ".join(f'{k.replace("_", "-")}="{esc(v)}"' for k, v in attrs.items())
    return f'<text x="{x}" y="{y}" class="{cls}" {extra}>{text}</text>'


def nice_ticks(max_val: float, n: int = 5) -> list[float]:
    if max_val <= 0:
        return [0, 1]
    raw = max_val / n
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    step = (1 if norm <= 1 else 2 if norm <= 2 else 5 if norm <= 5 else 10) * mag
    top = math.ceil(max_val / step) * step
    
    ticks = [i * step for i in range(int(round(top / step)) + 1)]
    return ticks if len(ticks) >= 2 else [0.0, max_val]


def format_value(v: float) -> str:
    if v == 0:
        return "0"
    if v < 0.001:
        return f"{v:.5f}".rstrip("0").rstrip(".")
    if v < 0.1:
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


def bar_svg(width_config: int, values: list[float], colors: list[str], ylabel: str, percent=False, fallback=None) -> str:
    labels = [x[0] for x in REGIONS]
    W, H = width_config, 240  
    ML, MR, MT, MB = 58, 12, 15, 35  
    CW, CH = W - ML - MR, H - MT - MB
    ticks = nice_ticks(max(values) * 1.15 if values else 1)
    ytop = max(ticks) if ticks else 1
    bw = CW / len(values) * 0.62  
    gap = CW / len(values)
    s = [f'<svg viewBox="0 0 {W} {H}" class="plot-svg">']
    s.append(f'<line x1="{ML}" y1="{MT+CH}" x2="{ML+CW}" y2="{MT+CH}" class="axis"/>')
    s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+CH}" class="axis"/>')
    for t in ticks:
        y = MT + CH - t / ytop * CH
        s.append(f'<line x1="{ML}" y1="{y:.2f}" x2="{ML+CW}" y2="{y:.2f}" class="grid"/>')
        label = f"{t:.0f}" if percent else format_value(t)
        s.append(svg_text(ML - 6, y + 4, esc(label), "tick", text_anchor="end"))
    for i, (v, lab, col) in enumerate(zip(values, labels, colors)):
        x = ML + i * gap + gap * 0.19
        bh = 0 if ytop == 0 else v / ytop * CH
        y = MT + CH - bh
        hatch_id = f"hatch-{i}".replace(" ", "")
        if fallback and fallback[i]:
            s.append(
                f'<defs><pattern id="{hatch_id}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                f'<rect width="6" height="6" fill="{col}"/><line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2"/></pattern></defs>'
            )
            fill = f"url(#{hatch_id})"
        else:
            fill = col
        s.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{bh:.2f}" fill="{fill}" stroke="#8a1f1f22"/>')
        val_txt = f"{v:.1f}%" if percent else format_value(v)
        s.append(svg_text(x + bw / 2, y - 5, esc(val_txt), "value", text_anchor="middle"))
        s.append(svg_text(x + bw / 2, MT + CH + 14, esc(lab), "xlab", text_anchor="end", transform=f"rotate(-45 {x+bw/2:.2f} {MT+CH+14})"))
    s.append(svg_text(16, MT + CH / 2, esc(ylabel), "axis-label", text_anchor="middle", transform=f"rotate(-90 16 {MT+CH/2:.2f})"))
    s.append("</svg>")
    return "\n".join(s)


def kde(vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.zeros_like(grid)
    if len(vals) == 1:
        bw = 0.08
    else:
        std = np.std(vals) or 0.1
        bw = max(1.06 * std * (len(vals) ** -0.2), 0.05)
    z = (grid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (len(vals) * bw * math.sqrt(2 * math.pi))
    return dens


def violin_svg(all_vals: list[np.ndarray], colors: list[str]) -> str:
    labels = [x[0] for x in REGIONS]
    W, H = 380, 240  
    ML, MR, MT, MB = 58, 12, 15, 35  
    CW, CH = W - ML - MR, H - MT - MB
    ymax = max([float(np.max(v)) for v in all_vals if len(v)] + [1.0])
    ticks = nice_ticks(ymax * 1.08, 5)
    ytop = max(ticks)
    grid = np.linspace(0, ytop, 120)
    rng = np.random.default_rng(7)
    gap = CW / len(all_vals)
    s = [f'<svg viewBox="0 0 {W} {H}" class="plot-svg">']
    s.append(f'<line x1="{ML}" y1="{MT+CH}" x2="{ML+CW}" y2="{MT+CH}" class="axis"/>')
    s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+CH}" class="axis"/>')
    for t in ticks:
        y = MT + CH - t / ytop * CH
        s.append(f'<line x1="{ML}" y1="{y:.2f}" x2="{ML+CW}" y2="{y:.2f}" class="grid"/>')
        s.append(svg_text(ML - 6, y + 4, esc(format_value(t)), "tick", text_anchor="end"))
    for i, vals in enumerate(all_vals):
        cx = ML + i * gap + gap / 2
        dens = kde(vals, grid)
        width = 28 * dens / max(dens.max(), 1e-9)
        left, right = [], []
        for g, w in zip(grid, width):
            y = MT + CH - g / ytop * CH
            left.append(f"{cx-w:.2f},{y:.2f}")
            right.append(f"{cx+w:.2f},{y:.2f}")
        s.append(f'<polygon points="{" ".join(left + right[::-1])}" fill="{colors[i]}" opacity="0.48" stroke="{colors[i]}" stroke-width="1"/>')
        if len(vals):
            sample = vals if len(vals) <= 300 else rng.choice(vals, 300, replace=False)
            for v in sample:
                y = MT + CH - v / ytop * CH
                x = cx + rng.normal(0, 9)
                s.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.25" fill="{colors[i]}" opacity="0.35"/>')
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            y1, ym, y3 = [MT + CH - q / ytop * CH for q in (q1, med, q3)]
            s.append(f'<rect x="{cx-7}" y="{y3:.2f}" width="14" height="{max(y1-y3,1):.2f}" fill="#fff" opacity="0.85" stroke="#222" stroke-width="1"/>')
            s.append(f'<line x1="{cx-8}" y1="{ym:.2f}" x2="{cx+8}" y2="{ym:.2f}" stroke="#111" stroke-width="1.5"/>')
        s.append(svg_text(cx, MT + CH + 14, esc(labels[i]), "xlab", text_anchor="end", transform=f"rotate(-45 {cx:.2f} {MT+CH+14})"))
    s.append(svg_text(16, MT + CH / 2, "log10(Density + 1)", "axis-label", text_anchor="middle", transform=f"rotate(-90 16 {MT+CH/2:.2f})"))
    s.append("</svg>")
    return "\n".join(s)


def venn_svg(counts: dict[str, int], colors: list[str]) -> str:
    W, H = 360, 240  
    circles = [
        (150, 95, 68, colors[0], "Promoter", counts["promoter_total"], 150, 18),
        (210, 95, 68, colors[2], "Gene Body", counts["body_total"], 255, 185),
        (180, 148, 68, "#9ca3af", "Intergenic", counts["intergenic_total"], 95, 185),
    ]
    pos = {
        "only_promoter": (125, 80),
        "only_body": (235, 80),
        "only_intergenic": (180, 188),
        "promoter_body": (180, 78),
        "promoter_intergenic": (148, 128),
        "body_intergenic": (212, 128),
        "all_three": (180, 112),
    }
    s = [f'<svg viewBox="0 0 {W} {H}" class="plot-svg venn-svg">']
    for cx, cy, r, col, lab, total, lx, ly in circles:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" opacity="0.38" stroke="{col}" stroke-width="2"/>')
        s.append(svg_text(lx, ly, esc(lab), "venn-label", text_anchor="middle"))
        s.append(svg_text(lx, ly + 12, f"{total:,}", "venn-total", text_anchor="middle"))
    for key, (x, y) in pos.items():
        if counts[key] > 0 or key == "all_three":
            s.append(svg_text(x, y, f"{counts[key]:,}", "venn-count", text_anchor="middle"))
    s.append("</svg>")
    legend = (
        f"<div class='venn-legend'>"
        f"Only Promoter: {counts['only_promoter']:,}<br>"
        f"Only Gene Body: {counts['only_body']:,}<br>"
        f"Only Intergenic: {counts['only_intergenic']:,}<br>"
        f"Promoter ∩ Body: {counts['promoter_body']:,} | "
        f"Promoter ∩ Inter: {counts['promoter_intergenic']:,}<br>"
        f"Body ∩ Inter: {counts['body_intergenic']:,} | "
        f"All three: {counts['all_three']:,}</div>"
    )
    return "\n".join(s) + legend


def pie_svg(title: str, values: list[int], labels: list[str], colors: list[str]) -> str:
    W, H = 300, 240  
    cx, cy, r = 150, 112, 64  
    total = sum(values)
    s = [f'<svg viewBox="0 0 {W} {H}" class="pie-svg">']
    s.append(svg_text(W / 2, 20, esc(title), "pie-title", text_anchor="middle"))
    if total == 0:
        s.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#eee" stroke="#fff"/>')
    else:
        start = -90
        for val, lab, col in zip(values, labels, colors):
            angle = 360 * val / total
            end = start + angle
            large = 1 if angle > 180 else 0
            x1 = cx + r * math.cos(math.radians(start))
            y1 = cy + r * math.sin(math.radians(start))
            x2 = cx + r * math.cos(math.radians(end))
            y2 = cy + r * math.sin(math.radians(end))
            s.append(f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} A {r} {r} 0 {large} 1 {x2:.2f} {y2:.2f} Z" fill="{col}" stroke="#fff" stroke-width="1.5"/>')
            mid = (start + end) / 2
            tx = cx + (r * 0.65) * math.cos(math.radians(mid))
            ty = cy + (r * 0.65) * math.sin(math.radians(mid))
            pct = val / total * 100
            if pct >= 7:
                s.append(svg_text(tx, ty - 3, esc(lab), "pie-label", text_anchor="middle"))
                s.append(svg_text(tx, ty + 10, f"{val:,}", "pie-label", text_anchor="middle"))
                s.append(svg_text(tx, ty + 23, f"({pct:.1f}%)", "pie-label", text_anchor="middle"))
            else:
                edge_x = cx + (r * 0.96) * math.cos(math.radians(mid))
                edge_y = cy + (r * 0.96) * math.sin(math.radians(mid))
                lx = cx + (r * 1.34) * math.cos(math.radians(mid))
                ly = cy + (r * 1.34) * math.sin(math.radians(mid))
                ly = min(max(ly, 40), H - 40)
                anchor = "start" if lx >= cx else "end"
                elbow_x = lx - 7 if lx >= cx else lx + 7
                s.append(f'<polyline points="{edge_x:.1f},{edge_y:.1f} {elbow_x:.1f},{ly:.1f} {lx:.1f},{ly:.1f}" fill="none" stroke="{col}" stroke-width="1"/>')
                s.append(svg_text(lx, ly - 2, esc(lab), "pie-label-dark", text_anchor=anchor))
                s.append(svg_text(lx, ly + 11, f"{val:,} ({pct:.1f}%)", "pie-label-dark-small", text_anchor=anchor))
            start = end
    s.append(svg_text(W / 2, 225, f"Total motifs = {total:,}", "pie-total", text_anchor="middle"))
    s.append("</svg>")
    return "\n".join(s)


def chromosome_separated_svg(labels: list[str], values: list[float], color: str, title: str) -> str:
    W, H = 260, 240
    ML, MR, MT, MB = 52, 14, 32, 44
    CW, CH = W - ML - MR, H - MT - MB
    ymax = max(values) * 1.15 if values else 1
    ticks = nice_ticks(ymax, 4)
    ytop = max(ticks) if max(ticks) > 0 else 1
    group = CW / len(labels)
    bw = group * 0.55
    
    s = [f'<svg viewBox="0 0 {W} {H}" class="chr-sub-svg">']
    s.append(svg_text(W / 2, 18, esc(title), "panel-subtitle", text_anchor="middle"))
    s.append(f'<line x1="{ML}" y1="{MT+CH}" x2="{ML+CW}" y2="{MT+CH}" class="axis"/>')
    s.append(f'<line x1="{ML}" y1="{MT}" x2="{ML}" y2="{MT+CH}" class="axis"/>')
    
    for t in ticks:
        y = MT + CH - t / ytop * CH
        s.append(f'<line x1="{ML}" y1="{y:.2f}" x2="{ML+CW}" y2="{y:.2f}" class="grid"/>')
        s.append(svg_text(ML - 6, y + 3, esc(format_value(t)), "tick", text_anchor="end"))
        
    for i, lab in enumerate(labels):
        cx = ML + i * group + group / 2
        h = values[i] / ytop * CH if ytop else 0
        x = cx - bw / 2
        y = MT + CH - h
        s.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw:.2f}" height="{h:.2f}" fill="{color}" stroke="#222" stroke-width="0.3"/>')
        s.append(svg_text(cx, y - 4, esc(format_value(values[i])), "value-small", text_anchor="middle"))
        s.append(svg_text(cx, MT + CH + 16, esc(lab), "xlab-flat", text_anchor="middle"))
        
    s.append(svg_text(16, MT + CH / 2, "Density per kb", "axis-label", text_anchor="middle", transform=f"rotate(-90 16 {MT+CH/2:.2f})"))
    s.append("</svg>")
    return "\n".join(s)


def heatmap_svg(title: str, matrix: pd.DataFrame, vmax: float, mode: str = "pis") -> str:
    rows, cols = matrix.shape
    cell_w, cell_h = 54, 14
    ML, MR, MT, MB = 118, 22, 114, 58  
    W = ML + cols * cell_w + MR
    H = MT + rows * cell_h + MB
    s = [f'<svg viewBox="0 0 {W} {H}" class="heatmap-svg">']
    s.append(svg_text(W / 2, 24, esc(title), "heat-title", text_anchor="middle"))
    
    for j, col in enumerate(matrix.columns):
        x = ML + j * cell_w + cell_w / 2
        label_y = MT - 22  
        s.append(svg_text(x, label_y, esc(col), "heat-x", text_anchor="middle", transform=f"rotate(-45 {x:.2f} {label_y})"))
        
    for i, (idx, row) in enumerate(matrix.iterrows()):
        y = MT + i * cell_h
        s.append(svg_text(ML - 6, y + cell_h * 0.72, esc(idx), "heat-y", text_anchor="end"))
        for j, val in enumerate(row.to_numpy(float)):
            x = ML + j * cell_w
            color = light_heatmap_hex(float(val), 0, vmax, mode=mode)
            s.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w:.2f}" height="{cell_h:.2f}" fill="{color}" stroke="#e2e8f0" stroke-width="0.5"/>')
            
            txt_color = "white" if val > vmax * 0.55 else "#111"
            s.append(
                f'<text x="{x + cell_w/2:.2f}" y="{y + cell_h*0.70:.2f}" '
                f'font-size="6.5" font-weight="700" fill="{txt_color}" text-anchor="middle">'
                f'{val:.2f}'
                f'</text>'
            )

    cb_x, cb_y, cb_w, cb_h = ML, H - 34, cols * cell_w, 10
    for k in range(120):
        x = cb_x + k * cb_w / 120
        color = light_heatmap_hex(vmax * k / 119, 0, vmax, mode=mode)
        s.append(f'<rect x="{x:.2f}" y="{cb_y}" width="{cb_w/120 + 0.3:.2f}" height="{cb_h}" fill="{color}"/>')
    s.append(f'<rect x="{cb_x}" y="{cb_y}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#94a3b8" stroke-width="0.5"/>')
    s.append(svg_text(cb_x, cb_y + 26, "0", "heat-cbar", text_anchor="middle"))
    s.append(svg_text(cb_x + cb_w, cb_y + 26, esc(f"{vmax:.2f}"), "heat-cbar", text_anchor="middle"))
    s.append(svg_text(cb_x + cb_w / 2, cb_y + 26, "log10(Density/kb + 1)", "heat-cbar", text_anchor="middle"))
    s.append("</svg>")
    return "\n".join(s)


def compute_scatter_metrics(df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 4000) -> dict | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None
        
    sub = df[[x_col, y_col]].dropna().copy()
    sub = sub[np.isfinite(sub[x_col]) & np.isfinite(sub[y_col])]
    sub = sub[sub[y_col] > 0]  
    
    if len(sub) < 5:
        return None
        
    x_vals = sub[x_col].to_numpy(dtype=float)
    y_raw = sub[y_col].to_numpy(dtype=float)
    y_log = np.log10(y_raw + 0.001)  
    
    try:
        r_p, p_p = pearsonr(x_vals, y_log)
        r_s, p_s = spearmanr(x_vals, y_log)
        slope, intercept = np.polyfit(x_vals, y_log, 1)
    except Exception:
        return None
        
    if len(sub) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sub), max_points, replace=False)
        x_vals = x_vals[idx]
        y_log = y_log[idx]
        
    points = [[float(x), float(yl)] for x, yl in zip(x_vals, y_log)]
    
    return {
        "points": points,
        "n": int(len(sub)),
        "r_pearson": float(r_p) if np.isfinite(r_p) else 0.0,
        "p_pearson": float(p_p) if np.isfinite(p_p) else 1.0,
        "r_spearman": float(r_s) if np.isfinite(r_s) else 0.0,
        "p_spearman": float(p_s) if np.isfinite(p_s) else 1.0,
        "slope": float(slope) if np.isfinite(slope) else 0.0,
        "intercept": float(intercept) if np.isfinite(intercept) else 0.0,
        "x_min": float(x_vals.min()),
        "x_max": float(x_vals.max()),
        "y_min": float(y_log.min()),
        "y_max": float(y_log.max())
    }


def export_top50_heatmap_files(pis_matrix: pd.DataFrame, pgs_matrix: pd.DataFrame, vmax: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 5.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 12), constrained_layout=True)
    fig.suptitle("Top 50 Gene Heatmaps", fontsize=16, fontweight="bold")

    cmap_pis = matplotlib.colors.LinearSegmentedColormap.from_list("light_pis", ["#fafafa", "#fdebc4", "#fcab9d", "#eb4b4b"])
    cmap_pgs = matplotlib.colors.LinearSegmentedColormap.from_list("light_pgs", ["#fafafa", "#e8f7eb", "#9dbea3", "#41aa50"])

    for ax, matrix, title, cmap in [
        (axes[0], pis_matrix, "A. Top 50 PIS-rich Genes", cmap_pis),
        (axes[1], pgs_matrix, "B. Top 50 PGS-rich Genes", cmap_pgs),
    ]:
        ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        ax.set_xlabel("Genomic region")
        ax.set_ylabel("wormbase_gene_id")
        ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#e2e8f0", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap_pis, norm=plt.Normalize(vmin=0, vmax=vmax))
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("log10(Density/kb + 1)")

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"Top50_Gene_Heatmaps.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def panel(title: str, body: str, row: str = "") -> str:
    return f"<section class='panel {row}'><div class='panel-title'>{title}</div>{body}</section>"


def build_html() -> str:
    pis_df = load_df(PIS_FILE)
    pgs_df = load_df(PGS_FILE)
    stats = load_stats()

    pis_mean, pis_fb = mean_density(pis_df, "pis", stats)
    pgs_mean, pgs_fb = mean_density(pgs_df, "pgs", stats)
    pis_pct = pct_values(pis_df, "pis")
    pgs_pct = pct_values(pgs_df, "pgs")
    chr_labels, pis_chr, pgs_chr = chr_density(stats)
    pis_heat = cluster_rows(top50_heatmap_matrix(pis_df, "pis"))
    pgs_heat = cluster_rows(top50_heatmap_matrix(pgs_df, "pgs"))
    heat_vmax = float(max(pis_heat.to_numpy().max(), pgs_heat.to_numpy().max()))

    composition_metrics = [
        ("GC%", "gc_percentage"),
        ("G%", "g_percentage"),
        ("C%", "c_percentage")  
    ]
    
    target_regions = [
        ("Promoter", "promoter"),
        ("5'UTR", "five_prime_UTR"),
        ("CDS", "CDS"),
        ("Intron", "intron"),
        ("3'UTR", "three_prime_UTR"),
        ("Gene Body", "gene_body")
    ]
    
    pis_scatter_matrix = []
    pgs_scatter_matrix = []
    
    for x_label, x_col in composition_metrics:
        for r_label, r_col in target_regions:
            pis_res = compute_scatter_metrics(pis_df, x_col, f"pis_{r_col}_density_per_kb")
            pgs_res = compute_scatter_metrics(pgs_df, x_col, f"pgs_{r_col}_density_per_kb")
            
            pis_scatter_matrix.append({
                "x_label": x_label, "region_label": r_label,
                "data": pis_res or {"points": [], "n": 0, "r_pearson": 0, "p_pearson": 1, "r_spearman": 0, "p_spearman": 1, "slope": 0, "intercept": 0}
            })
            pgs_scatter_matrix.append({
                "x_label": x_label, "region_label": r_label,
                "data": pgs_res or {"points": [], "n": 0, "r_pearson": 0, "p_pearson": 1, "r_spearman": 0, "p_spearman": 1, "slope": 0, "intercept": 0}
            })

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Genomic Motif Atlas - C. elegans WBcel235</title>
<style>
:root {{
  --blue: {BLUE};
  --pis: #d60000;
  --pgs: #08752f;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: #fff;
  color: #111;
  font-family: Arial, Helvetica, sans-serif;
}}
.sheet {{
  width: 100%; 
  max-width: 1600px; 
  margin: 0 auto;
  background: #fff;
  padding: 10px 18px 18px; 
}}
.master-title {{
  background: white;
  color: #111;
  font-weight: 850;
  text-align: center;
  letter-spacing: .3px;
  font-size: 24px;
  padding: 12px 12px 24px;
  text-transform: uppercase;
}}
.figure2 {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px; 
  border-left: 0;
  border-right: 0;
}}
.panel {{
  min-height: 275px;
  padding: 12px 4px 6px;
  border: 0;
  background: none;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
}}
.panel.pgs {{
  border-color: transparent;
}}
.panel-title {{
  color: #111;
  font-weight: 800;
  font-size: 16px;
  text-align: center;
  margin: 0 0 12px;
}}
.panel.pgs .panel-title {{ color: #111; }}
.bottom {{
  display: grid;
  grid-template-columns: 1fr 1fr 1.35fr;
  gap: 18px;
  margin-top: 24px; 
}}
.bottom-block {{
  border: 0;
  background: none;
  min-height: 300px; 
}}
.bottom-title {{
  background: white;
  color: #111;
  font-weight: 850;
  text-align: center;
  font-size: 18px;
  padding: 8px 6px 2px;
  text-transform: uppercase;
}}
.pie-row {{
  display: flex;
  justify-content: space-around;
  align-items: flex-start;
  padding: 8px 12px 6px;
  gap: 20px;
}}
.chr-split-row {{
  display: flex;
  justify-content: space-around;
  align-items: center;
  padding: 14px 8px;
  gap: 10px;
}}
.plot-svg {{ width: 100%; height: auto; overflow: visible; }}
.pie-svg {{ width: 49%; height: auto; overflow: visible; }}
.chr-sub-svg {{ width: 48%; height: auto; overflow: visible; }}
.panel-subtitle {{ font-weight: 800; font-size: 15px; fill: currentColor; }}
.axis {{ stroke: #111; stroke-width: 1.1; }}
.grid {{ stroke: #e2e8f0; stroke-width: .8; }}
.tick {{ font-size: 11px; fill: #111; font-weight: 600; }}
.xlab {{ font-size: 11px; fill: #111; font-weight: 650; }}
.xlab-flat {{ font-size: 12px; fill: #111; font-weight: 650; }}
.axis-label {{ font-size: 13px; fill: #111; font-weight: 800; }}
.value {{ font-size: 10px; fill: #111; font-weight: 800; }}
.value-small {{ font-size: 9.5px; fill: #111; font-weight: 750; }}
.venn-label {{ font-size: 13px; font-weight: 800; fill: #25406f; }}
.venn-total {{ font-size: 11px; font-weight: 800; fill: #111; }}
.venn-count {{ font-size: 11px; font-weight: 850; fill: #111; }}
.venn-legend {{
  font-size: 10.5px;
  line-height: 1.25;
  text-align: center;
  border: 1px solid #cfcfcf;
  background: rgba(255,255,255,.86);
  padding: 5px 6px;
  width: 260px;
  margin-top: 4px;
}}
.pie-title {{ font-size: 14px; font-weight: 850; }}
.pie-label {{ font-size: 10px; font-weight: 800; fill: white; }}
.pie-label-dark {{ font-size: 9px; font-weight: 800; fill: #111; }}
.pie-label-dark-small {{ font-size: 8px; font-weight: 700; fill: #111; }}
.pie-total {{ font-size: 10px; font-weight: 800; fill: #111; }}
.legend-text {{ font-size: 12px; font-weight: 800; }}
.footer {{
  font-size: 13px;
  display: flex;
  justify-content: center; 
  text-align: center; 
  padding: 0 20px 0; 
  margin-top: -10px; 
}}
.figure6 {{
  margin-top: 4px; 
  padding-top: 10px;
  border-top: 0;
  background: none; 
}}
.heatmap-row {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 28px;
  align-items: start;
}}
.heatmap-svg {{
  width: 100%;
  height: auto;
  overflow: visible;
}}
.heat-title {{ font-size: 16px; font-weight: 850; fill: #111; }}
.heat-x {{ font-size: 10px; font-weight: 750; fill: #111; }}
.heat-y {{ font-size: 7.2px; fill: #111; }}
.heat-cbar {{ font-size: 10px; font-weight: 650; fill: #111; }}

.figure7 {{
  margin-top: 32px;
  padding-top: 20px;
  border-top: 2px solid #e2e8f0;
}}
.section-headline {{
  font-size: 20px;
  font-weight: 850;
  text-transform: uppercase;
  margin-bottom: 16px;
  color: #111;
  text-align: center;
}}
.scatter-grid {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 24px;
}}
.scatter-panel {{
  background: #fafafa;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  padding: 6px;
}}
.scatter-title {{
  font-size: 11px;
  font-weight: 800;
  text-align: center;
  margin-bottom: 4px;
  color: #333;
}}
.sc-svg {{
  width: 100%;
  height: auto;
  overflow: visible;
}}
.sc-axis {{ stroke: #222; stroke-width: 1; }}
.sc-grid {{ stroke: #e2e8f0; stroke-width: 0.6; }}
.sc-tick {{ font-size: 8.5px; fill: #444; }}
.sc-label {{ font-size: 10px; font-weight: 700; fill: #111; }}
.sc-stats {{ font-size: 8px; fill: #222; font-family: monospace; font-weight: bold; }}

@media print {{
  body {{ background: white; }}
  .sheet {{ width: 100%; }}
}}
</style>
</head>
<body>
<main class="sheet">
  <div class="master-title">Genomic analysis of Caenorhabditis Elegans (WBcel235)</div>

  <div class="figure2">
    {panel("PIS Mean Density", bar_svg(360, pis_mean, PIS, "Mean Density (per kb)", fallback=pis_fb))}
    {panel("PIS Density Distribution", violin_svg(dist_values(pis_df, "pis", stats), PIS))}
    {panel("PIS % Genes with Motif", bar_svg(360, pis_pct, PIS, "% Genes", percent=True))}
    {panel("PIS Region Overlap", venn_svg(overlap3(pis_df, "pis"), ["#fb7185", "#f97316", "#93c5fd"]))}

    {panel("PGS Mean Density", bar_svg(360, pgs_mean, PGS, "Mean Density (per kb)", fallback=pgs_fb), "pgs")}
    {panel("PGS Density Distribution", violin_svg(dist_values(pgs_df, "pgs", stats), PGS), "pgs")}
    {panel("PGS % Genes with Motif", bar_svg(360, pgs_pct, PGS, "% Genes", percent=True), "pgs")}
    {panel("PGS Region Overlap", venn_svg(overlap3(pgs_df, "pgs"), ["#4ade80", "#84cc16", "#93c5fd"]), "pgs")}
  </div>

  <div class="bottom">
    <section class="bottom-block">
      <div class="bottom-title">Exon Composition <span style="font-size:14px">(Exonic Motifs)</span></div>
      <div class="pie-row">
        {pie_svg("PIS (Exonic Motifs)", sum_counts(pis_df, "pis", EXONIC), [x[0] for x in EXONIC], [PIS[0], PIS[1], PIS[2]])}
        {pie_svg("PGS (Exonic Motifs)", sum_counts(pgs_df, "pgs", EXONIC), [x[0] for x in EXONIC], [PGS[0], PGS[1], PGS[2]])}
      </div>
    </section>

    <section class="bottom-block">
      <div class="bottom-title">Gene Body Composition <span style="font-size:14px">(Within Genes)</span></div>
      <div class="pie-row">
        {pie_svg("PIS (Gene Body Motifs)", sum_counts(pis_df, "pis", GENE_BODY), [x[0] for x in GENE_BODY], [PIS[0], PIS[1], "#5f8bc9", PIS[2]])}
        {pie_svg("PGS (Gene Body Motifs)", sum_counts(pgs_df, "pgs", GENE_BODY), [x[0] for x in GENE_BODY], [PGS[0], PGS[1], "#5f8bc9", PGS[2]])}
      </div>
    </section>

    <section class="bottom-block">
      <div class="bottom-title">Chromosome-wise Motif Density <span style="font-size:14px">(per kb)</span></div>
      <div class="chr-split-row">
        {chromosome_separated_svg(chr_labels, pis_chr, "#ef4444", "PIS Chromosome Density")}
        {chromosome_separated_svg(chr_labels, pgs_chr, "#4caf50", "PGS Chromosome Density")}
      </div>
    </section>
  </div>

  <div class="footer">
    <div>Densities normalized per kb where exported; hatched promoter/intergenic bars use data-derived proxy densities because catalogue density columns were not exported.</div>
  </div>

  <section class="figure6">
    <div class="heatmap-row">
      {heatmap_svg("Top 50 PIS-rich Genes", pis_heat, heat_vmax, mode="pis")}
      {heatmap_svg("Top 50 PGS-rich Genes", pgs_heat, heat_vmax, mode="pgs")}
    </div>
    <div class="figure-caption" style="text-align: center; font-size: 12px; margin-top: 6px; font-weight: bold;">
      Rows are clustered hierarchically; columns remain fixed in biological order. Both heatmaps use independent light-spectrum linear colormaps keyed to log10(density/kb + 1).
    </div>
  </section>

  <section class="figure7">
    <div class="section-headline">PIS Motif Composition Correlation Matrix</div>
    <div id="pisScatterGrid" class="scatter-grid"></div>
    
    <div class="section-headline" style="margin-top: 40px;">PGS Motif Composition Correlation Matrix</div>
    <div id="pgsScatterGrid" class="scatter-grid"></div>
  </section>
</main>

<script>
const PIS_SCATTERS = {json.dumps(pis_scatter_matrix)};
const PGS_SCATTERS = {json.dumps(pgs_scatter_matrix)};

function drawScatter(item, dotColor) {{
    const d = item.data;
    if (!d || !d.points || d.points.length === 0) {{
        return `<div class="scatter-panel">
            <div class="scatter-title">\${{item.x_label}} vs \${{item.region_label}}</div>
            <div style="height:130px; display:flex; align-items:center; justify-content:center; font-size:10px; color:#aaa; font-weight:bold;">No Active Data</div>
        </div>`;
    }}
    
    const W = 220, H = 180;
    const ML = 38, MR = 12, MT = 12, MB = 32;
    const CW = W - ML - MR, CH = H - MT - MB;
    
    let xMin = d.x_min, xMax = d.x_max;
    let yMin = d.y_min, yMax = d.y_max;
    if (xMin === xMax) {{ xMin -= 1; xMax += 1; }}
    if (yMin === yMax) {{ yMin -= 0.5; yMax += 0.5; }}
    
    xMin -= (xMax - xMin) * 0.05; xMax += (xMax - xMin) * 0.05;
    yMin -= (yMax - yMin) * 0.05; yMax += (yMax - yMin) * 0.05;
    
    let s = [\`<svg viewBox="0 0 \${{W}} \${{H}}" class="sc-svg">\`];
    s.push(\`<line x1="\${{ML}}" y1="\${{MT+CH}}" x2="\${{ML+CW}}" y2="\${{MT+CH}}" class="sc-axis"/>\`);
    s.push(\`<line x1="\${{ML}}" y1="\${{MT}}" x2="\${{ML}}" y2="\${{MT+CH}}" class="sc-axis"/>\`);
    
    const xTicks = [xMin + (xMax-xMin)*0.1, xMin + (xMax-xMin)*0.5, xMin + (xMax-xMin)*0.9];
    xTicks.forEach(t => {{
        const x = ML + ((t - xMin) / (xMax - xMin)) * CW;
        s.push(\`<line x1="\${{x}}" y1="\${{MT}}" x2="\${{x}}" y2="\${{MT+CH}}" class="sc-grid"/>\`);
        s.push(\`<text x="\${{x}}" y="\${{MT+CH+12}}" class="sc-tick" text-anchor="middle">\${{t.toFixed(1)}}</text>\`);
    }});
    
    const yTicks = [yMin + (yMax-yMin)*0.1, yMin + (yMax-yMin)*0.5, yMin + (yMax-yMin)*0.9];
    yTicks.forEach(t => {{
        const y = MT + CH - ((t - yMin) / (yMax - yMin)) * CH;
        s.push(\`<line x1="\${{ML}}" y1="\${{y}}" x2="\${{ML+CW}}" y2="\${{y}}" class="sc-grid"/>\`);
        s.push(\`<text x="\${{ML-4}}" y="\${{y+3}}" class="sc-tick" text-anchor="end">\${{t.toFixed(2)}}</text>\`);
    }});
    
    d.points.forEach(pt => {{
        const cx = ML + ((pt[0] - xMin) / (xMax - xMin)) * CW;
        const cy = MT + CH - ((pt[1] - yMin) / (yMax - yMin)) * CH;
        s.push(\`<circle cx="\${{cx.toFixed(1)}}" cy="\${{cy.toFixed(1)}}" r="1.5" fill="\${{dotColor}}" opacity="0.4"/>\`);
    }});
    
    const y1_val = d.slope * xMin + d.intercept;
    const y2_val = d.slope * xMax + d.intercept;
    const ly1 = MT + CH - ((y1_val - yMin) / (yMax - yMin)) * CH;
    const ly2 = MT + CH - ((y2_val - yMin) / (yMax - yMin)) * CH;
    
    if (!isNaN(ly1) && !isNaN(ly2)) {{
        s.push(\`<line x1="\${{ML.toFixed(1)}}" y1="\${{ly1.toFixed(1)}}" x2="\${{(ML+CW).toFixed(1)}}" y2="\${{ly2.toFixed(1)}}" stroke="#222" stroke-width="1.5" stroke-dasharray="2,2"/>\`);
    }}
    
    const pSign = d.p_pearson < 0.001 ? "p<0.001" : \`p=\${{d.p_pearson.toFixed(3)}}\`;
    s.push(\`<text x="\${{ML+6}}" y="\${{MT+14}}" class="sc-stats">R=\${{d.r_pearson.toFixed(2)}}</text>\`);
    s.push(\`<text x="\${{ML+6}}" y="\${{MT+24}}" class="sc-stats">\${{pSign}}</text>\`);
    s.push(\`<text x="\${{ML+6}}" y="\${{MT+34}}" class="sc-stats">n=\${{d.n}}</text>\`);
    
    s.push(\`<text x="\${{ML + CW/2}}" y="\${{H-4}}" class="sc-label" text-anchor="middle">\${{item.x_label}}</text>\`);
    s.push(\`<text x="10" y="\${{MT + CH/2}}" class="sc-label" text-anchor="middle" transform="rotate(-90 10 \${{MT + CH/2}})">log10(Density)</text>\`);
    s.push(\`</svg>\`);
    
    return \`
    <div class="scatter-panel">
        <div class="scatter-title">\${{item.x_label}} vs \${{item.region_label}}</div>
        \${{s.join('\\n')}}
    </div>\`;
}}

function renderScatterGrid(targetId, matrix, dotColor) {{
    const container = document.getElementById(targetId);
    container.innerHTML = matrix.map(item => drawScatter(item, dotColor)).join("");
}}

document.addEventListener("DOMContentLoaded", () => {{
    renderScatterGrid("pisScatterGrid", PIS_SCATTERS, "#ef4444");
    renderScatterGrid("pgsScatterGrid", PGS_SCATTERS, "#41aa50");
}});
</script>
</body>
</html>
"""
    return html_doc


def main() -> None:
    pis_df = load_df(PIS_FILE)
    pgs_df = load_df(PGS_FILE)
    pis_heat = cluster_rows(top50_heatmap_matrix(pis_df, "pis"))
    pgs_heat = cluster_rows(top50_heatmap_matrix(pgs_df, "pgs"))
    heat_vmax = float(max(pis_heat.to_numpy().max(), pgs_heat.to_numpy().max()))
    export_top50_heatmap_files(pis_heat, pgs_heat, heat_vmax)

    out = OUT_DIR / "Genomic_Motif_Atlas.html"
    out.write_text(build_html(), encoding="utf-8")
    print(f"Created: {out}")
    print(f"Created: {OUT_DIR / 'Top50_Gene_Heatmaps.png'}")
    print(f"Created: {OUT_DIR / 'Top50_Gene_Heatmaps.svg'}")
    print(f"Created: {OUT_DIR / 'Top50_Gene_Heatmaps.pdf'}")


if __name__ == "__main__":
    main()