import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# Load hit tables
# ============================================================

pgs = pd.read_csv(
    "/Users/chandanmehta/Desktop/G4_Project/DNA_Pipeline/results/PGS_Hits_DNA_Celegans_WBcel235.csv"
)

pis = pd.read_csv(
    "/Users/chandanmehta/Desktop/G4_Project/DNA_Pipeline/results/PIS_Hits_DNA_Celegans_WBcel235.csv"
)

pgs["Motif"] = "PGS"
pis["Motif"] = "PIS"

df = pd.concat([pgs, pis], ignore_index=True)

# ============================================================
# FIGURE 1
# Broad genomic regions
# promoter / TSS / gene body 
# ============================================================

broad_counts = (
    df.groupby(
        ["Motif", "broad_region", "wormbase_gene_id"]
    )
    .size()
    .reset_index(name="Motifs_per_Gene")
)

region_order_broad = [
    "promoter",
    "TSS_region",
    "gene_body",
]

plt.figure(figsize=(9, 6))

sns.violinplot(
    data=broad_counts,
    x="broad_region",
    y="Motifs_per_Gene",
    hue="Motif",
    order=region_order_broad,
    cut=0,
    inner="quartile"
)

plt.yscale("log")
plt.xlabel("Broad Genomic Region")
plt.ylabel("Number of Motifs per Gene")
plt.title("G4 and i-Motif Distribution Across Broad Genomic Regions")

plt.tight_layout()

plt.savefig(
    "Broad_Genomic_Region_Violin.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FIGURE 2
# Gene architecture
# Only motifs inside gene_body
# ============================================================

gene_body_df = df[
    df["broad_region"] == "gene_body"
].copy()

fine_counts = (
    gene_body_df.groupby(
        ["Motif", "fine_region", "wormbase_gene_id"]
    )
    .size()
    .reset_index(name="Motifs_per_Gene")
)

region_order_fine = [
    "five_prime_UTR",
    "CDS",
    "intron",
    "three_prime_UTR"
]

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=fine_counts,
    x="fine_region",
    y="Motifs_per_Gene",
    hue="Motif",
    order=region_order_fine,
    cut=0,
    inner="quartile"
)

plt.yscale("log")
plt.xlabel("Gene Feature")
plt.ylabel("Number of Motifs per Gene")
plt.title("G4 and i-Motif Distribution Within Gene Bodies")

plt.tight_layout()

plt.savefig(
    "Gene_Feature_Violin.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# Summary statistics
# ============================================================

print("\n=== Broad Regions ===")
print(
    broad_counts.groupby("broad_region")["Motifs_per_Gene"]
    .describe()
)

print("\n=== Gene Features ===")
print(
    fine_counts.groupby("fine_region")["Motifs_per_Gene"]
    .describe()
)