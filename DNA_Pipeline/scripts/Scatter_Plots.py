from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path("/Users/chandanmehta/Desktop/G4_Project/DNA_Pipeline/results")

PIS_FILE = BASE_DIR / "PIS_Catalogue_DNA_Celegans_WBcel235.csv"
PGS_FILE = BASE_DIR / "PGS_Catalogue_DNA_Celegans_WBcel235.csv"

OUT_DIR = BASE_DIR / "DNA_Scatter_Plots"
OUT_DIR.mkdir(exist_ok=True)

# =====================================================
# Plot settings
# =====================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLORS = {
    "PIS": "#d62728",
    "PGS": "#2ca02c"
}

# Exactly 5 regions (Promoter completely removed)
REGIONS = [
    ("5'UTR", "five_prime_UTR"),
    ("CDS", "CDS"),
    ("Intron", "intron"),
    ("3'UTR", "three_prime_UTR"),
    ("Gene Body", "gene_body")
]

X_COLUMNS = [
    ("GC %", "gc_percentage"),
    ("G %", "g_percentage"),
    ("C %", "c_percentage")
]


# =====================================================
# Gene body density
# =====================================================

def add_gene_body_density(df, prefix):
    count = (
        df[f"{prefix}_five_prime_UTR_count"] +
        df[f"{prefix}_CDS_count"] +
        df[f"{prefix}_intron_count"] +
        df[f"{prefix}_three_prime_UTR_count"]
    )

    df[f"{prefix}_gene_body_density_per_kb"] = (
        count / df["gene_length"] * 1000
    ).replace([np.inf, -np.inf], np.nan)

    return df


# =====================================================
# Generate Consolidated Panel Plot
# =====================================================

def generate_panel(df, motif):
    print(f"Generating clean 15-panel layout for {motif}...")
    
    df = add_gene_body_density(df, motif.lower())
    
    # 3x5 layout grid config
    fig, axes = plt.subplots(
        nrows=3,
        ncols=5,
        figsize=(19, 12),
        sharey="row"
    )
    
    color = COLORS[motif]
    
    # structural loop mapping over 15 axes
    for row, (xlabel, xcol) in enumerate(X_COLUMNS):
        for col, (region, reg) in enumerate(REGIONS):
            ax = axes[row, col]
            
            density_col = f"{motif.lower()}_{reg}_density_per_kb"
            
            # Filter valid points
            tmp = df[[xcol, density_col]].dropna()
            tmp = tmp[tmp[density_col] > 0]
            
            # Region header titles applied to every subplot loop iteration
            ax.set_title(region, fontsize=14, fontweight="bold")
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Handle edge cases where data points are insufficient 
            if len(tmp) < 5:
                ax.text(
                    0.5, 0.5,
                    "n < 5",
                    ha="center",
                    va="center",
                    fontsize=12
                )
                if row < 2:
                    ax.tick_params(labelbottom=False)
                continue
                
            x = tmp[xcol].values
            y = np.log10(tmp[density_col].values + 0.001)
            
            pear_r, pear_p = pearsonr(x, y)
            spear_r, spear_p = spearmanr(x, y)
            
            m, b = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            yy = m * xx + b
            
            ax.scatter(
                x, y,
                s=12,
                alpha=0.45,
                color=color,
                edgecolors="none"
            )
            ax.plot(xx, yy, color="black", lw=2)
            
            # Statistical box template on all active plots
            stats = (
                f"r = {pear_r:.2f}\n"
                f"ρ = {spear_r:.2f}\n"
                f"p = {pear_p:.2e}\n"
                f"n = {len(x)}"
            )
            ax.text(
                0.03, 0.97,
                stats,
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                bbox=dict(
                    facecolor="white",
                    edgecolor="gray",
                    alpha=0.9
                )
            )
            
            # Configure X-axis tick display for row layout depth
            if row == 2:
                ax.set_xlabel(f"{xlabel} (%)")
            else:
                ax.tick_params(labelbottom=False)

    # Global Figure Annotations
    fig.suptitle(
        f"{motif} Density vs GC%, G% and C%",
        fontsize=22,
        fontweight="bold",
        y=0.985
    )
    
    # Matrix margin annotations 
    fig.text(0.02, 0.82, "GC %", rotation=90, fontsize=18, fontweight="bold", va="center")
    fig.text(0.02, 0.50, "G %", rotation=90, fontsize=18, fontweight="bold", va="center")
    fig.text(0.02, 0.18, "C %", rotation=90, fontsize=18, fontweight="bold", va="center")
    
    fig.text(
        0.005, 0.5,
        r"log$_{10}$(Density / kb + 0.001)",
        rotation=90,
        fontsize=18,
        va="center"
    )
    
    fig.text(
        0.5, 0.02,
        "Nucleotide Percentage (%)",
        ha="center",
        fontsize=18,
        fontweight="bold"
    )
    
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.08,
        wspace=0.15,
        hspace=0.15
    )
    
    plt.savefig(OUT_DIR / f"{motif}_Correlation_Panel.png", dpi=600)
    plt.savefig(OUT_DIR / f"{motif}_Correlation_Panel.pdf")
    plt.savefig(OUT_DIR / f"{motif}_Correlation_Panel.svg")
    plt.close()


# =====================================================
# Main
# =====================================================

def main():
    pis = pd.read_csv(PIS_FILE)
    pgs = pd.read_csv(PGS_FILE)

    generate_panel(pis, "PIS")
    generate_panel(pgs, "PGS")

    print("\nDone. 15-panel figures generated with promoter profiles completely removed.\n")


if __name__ == "__main__":
    main()