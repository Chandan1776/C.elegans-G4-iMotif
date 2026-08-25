#!/opt/anaconda3/envs/g4_project/bin/python
"""
========================================================================
PIS & PGS mRNA Catalogue — C. elegans WBcel235 (Ensembl release 113)
========================================================================
Scans the entire C. elegans transcriptome for:
  - PIS  (i-Motif):        C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}
  - PGS  (G-Quadruplex):   G{2,5}[ATCG]{1,7}G{2,5}[ATCG]{1,7}G{2,5}[ATCG]{1,7}G{2,5}

Reference files (Ensembl release 113, assembly WBcel235):
  FASTA : Caenorhabditis_elegans.WBcel235.cdna.all.fa.gz
  GTF   : Caenorhabditis_elegans.WBcel235.113.gtf.gz

Key differences from GENCODE/human pipeline:
  - FASTA header: Ensembl space-delimited key:value format
      >T19E7.2.1 gene:WBGene00020878 gene_biotype:protein_coding
       transcript_biotype:protein_coding gene_symbol:T19E7.2 ...
  - Gene IDs: WBGene00000001 format
  - Gene names: unc-22, daf-2, etc. (WormBase names)
  - Transcript IDs: F13H10.4a.1, T19E7.2.1, etc. (no ENST prefix)
  - GTF attribute keys: gene_id, gene_name, transcript_id, transcript_name,
                        gene_biotype, transcript_biotype
  - Writes to /tmp staging dir first to avoid Windows/OneDrive PermissionError

Outputs (saved to OUTPUT_DIR):
  PGS_Hits_Celegans_WBcel235.csv
  PGS_Catalogue_Celegans_WBcel235.csv
  PGS_Summary_Celegans_WBcel235.csv
  PIS_Hits_Celegans_WBcel235.csv
  PIS_Catalogue_Celegans_WBcel235.csv
  PIS_Summary_Celegans_WBcel235.csv
  PIS_PGS_Summary_Celegans_WBcel235.html

Usage:
  python3 scan_Celegans_WBcel235_DNA.py \
      --fasta  "/path/to/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz" \
      --gff3   "/path/to/caenorhabditis_elegans.PRJNA13758.WBPS19.annotations.gff3.gz" \
      --outdir "/path/to/output/"
========================================================================
"""

from html import parser
import re
import sys
import csv
import gzip
import json
import shutil
import time
import tempfile
import argparse
import logging
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

def load_functional_descriptions(desc_file):
    """
    Parse WormBase functional_descriptions.txt(.gz)

    Returns:
        dict: WBGeneID -> functional description
    """
    import gzip

    descriptions = {}

    opener = gzip.open if str(desc_file).endswith(".gz") else open

    with opener(desc_file, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):

        line = lines[i].strip()

        if not line or line.startswith("#"):
            i += 1
            continue

        if line.startswith("WBGene"):

            gene_id = line.split("\t")[0]

            concise_lines = []
            automated_lines = []

            j = i + 1
            mode = None

            while j < len(lines):

                current = lines[j].rstrip()

                if current.strip() == "=":
                    break

                if current.startswith("Concise description:"):
                    mode = "concise"
                    concise_lines.append(
                    current.replace(
                        "Concise description:", ""
                    ).strip()
                    )

                elif current.startswith("Automated description:"):
                    mode = "automated"
                    automated_lines.append(
                    current.replace("Automated description:",   ""
                ).strip()
                )

                else:

                    if mode == "concise":
                        concise_lines.append(current.strip())

                    elif mode == "automated":
                        automated_lines.append(current.strip())

                j += 1

            concise_text = " ".join(concise_lines).strip()
            automated_text = " ".join(automated_lines).strip()

            if concise_text:
                descriptions[gene_id] = concise_text
            else:
                descriptions[gene_id] = automated_text

            i = j 

        i += 1

    return descriptions

def load_protein_domains(domain_file):
    """
    Parse WormBase protein_domains.tsv

    Returns:
        dict: WBGeneID -> protein domains
    """

    domains = {}

    with open(domain_file, "r", encoding="utf-8", errors="replace") as fh:

        for line in fh:

            line = line.rstrip()

            if not line:
                continue

            fields = line.split("\t")

            gene_id = fields[0]

            domain_names = []

            for item in fields[3:]:

                if '"' in item:

                    try:
                        domain_names.append(
                            item.split('"')[1]
                        )
                    except Exception:
                        pass

            if gene_id not in domains:
                domains[gene_id] = set()

            domains[gene_id].update(domain_names)

    return {
        gid: "; ".join(sorted(vals))
        for gid, vals in domains.items()
    }


# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Motif patterns (identical to PIS_PGS_mRNA_Scanner.html) ──────────────────
PIS_PATTERN = re.compile(r'C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}')
PGS_PATTERN = re.compile(r'G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}')

SPECIES_TAG = 'Celegans_WBcel235'

# ── Column definitions ────────────────────────────────────────────────────────
HITS_FIELDNAMES = [
    'wormbase_gene_id', 'gene_symbol', 'gene_function', 'protein_domains', 'transcript_id', 'transcript_name',
    'biotype', 'chromosome', 'strand',
    'transcript_length', 'utr5_length', 'cds_length', 'utr3_length',
    'motif_type', 'region',
    'cdna_start', 'cdna_end',
    'local_start', 'local_end',
    'motif_length', 'sequence',
    'tract_lengths', 'loop_lengths',
    'region_length', 'density_per_kb',
]

CATALOGUE_FIELDNAMES = [
    'wormbase_gene_id', 'gene_symbol', 'gene_function', 'protein_domains', 'transcript_id', 'transcript_name',
    'biotype', 'chromosome', 'strand',
    'transcript_length',
    'g_count', 'c_count', 'a_count', 't_count',
    'g_percentage', 'c_percentage', 'gc_percentage',
    'utr5_length', 'cds_length', 'utr3_length',
    # PIS
    'pis_utr5_count', 'pis_cds_count', 'pis_utr3_count', 'pis_total',
    'pis_utr5_density_per_kb', 'pis_cds_density_per_kb',
    'pis_utr3_density_per_kb', 'pis_total_density_per_kb',
    'pis_utr5_locations', 'pis_cds_locations', 'pis_utr3_locations',
    'pis_utr5_sequences', 'pis_cds_sequences', 'pis_utr3_sequences',
    # PGS
    'pgs_utr5_count', 'pgs_cds_count', 'pgs_utr3_count', 'pgs_total',
    'pgs_utr5_density_per_kb', 'pgs_cds_density_per_kb',
    'pgs_utr3_density_per_kb', 'pgs_total_density_per_kb',
    'pgs_utr5_locations', 'pgs_cds_locations', 'pgs_utr3_locations',
    'pgs_utr5_sequences', 'pgs_cds_sequences', 'pgs_utr3_sequences',
]

SUMMARY_FIELDNAMES = [
    'wormbase_gene_id', 'gene_symbol', 'gene_function', 'protein_domains', 'chromosome', 'strand', 'biotype',
    'n_transcripts',
    'pis_total_hits', 'pis_transcripts_with_any',
    'pis_transcripts_with_utr5', 'pis_transcripts_with_cds', 'pis_transcripts_with_utr3',
    'pis_pct_transcripts_with_any',
    'pis_pct_transcripts_with_utr5', 'pis_pct_transcripts_with_cds', 'pis_pct_transcripts_with_utr3',
    'pis_avg_density_utr5', 'pis_avg_density_cds', 'pis_avg_density_utr3',
    'pis_avg_total_density',
    'pgs_total_hits', 'pgs_transcripts_with_any',
    'pgs_transcripts_with_utr5', 'pgs_transcripts_with_cds', 'pgs_transcripts_with_utr3',
    'pgs_pct_transcripts_with_any',
    'pgs_pct_transcripts_with_utr5', 'pgs_pct_transcripts_with_cds', 'pgs_pct_transcripts_with_utr3',
    'pgs_avg_density_utr5', 'pgs_avg_density_cds', 'pgs_avg_density_utr3',
    'pgs_avg_total_density',
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Parse Ensembl GTF
# ══════════════════════════════════════════════════════════════════════════════

def parse_gtf(gtf_path: Path) -> dict:
    """
    Parse Ensembl GTF (WBcel235) for per-transcript region lengths.

    Ensembl GTF attribute format (same key "value" syntax as GENCODE):
      gene_id "WBGene00000001"; gene_version "6"; transcript_id "B0001.1";
      gene_name "aex-3"; transcript_name "aex-3-201";
      gene_biotype "protein_coding"; transcript_biotype "protein_coding";

    Returns:
      { transcript_id: {
          'gene_id', 'gene_name', 'transcript_name', 'biotype',
          'chrom', 'strand', 'utr5_len', 'cds_len', 'utr3_len'
      }}
    """
    log.info(f"Parsing GTF: {gtf_path}")

    raw = defaultdict(lambda: {
        'gene_id': '', 'gene_name': '', 'transcript_name': '',
        'biotype': '', 'chrom': '', 'strand': '',
        'utr5': [], 'cds': [], 'utr3': [],
    })

    opener = gzip.open if str(gtf_path).endswith('.gz') else open
    n_lines = 0
    with opener(gtf_path, 'rt', errors='replace') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            n_lines += 1
            if n_lines % 1_000_000 == 0:
                log.info(f"  … {n_lines:,} GTF lines")
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9:
                continue
            feature = cols[2]
            if feature not in ('five_prime_utr', 'three_prime_utr', 'CDS', 'transcript'):
                continue

            chrom    = cols[0]
            start    = int(cols[3])
            end      = int(cols[4])
            strand   = cols[6]
            attr_str = cols[8]

            # Parse key "value"; attribute pairs
            attrs = {}
            for m in re.finditer(r'(\w+)\s+"([^"]+)"', attr_str):
                attrs[m.group(1)] = m.group(2)

            tx_id = attrs.get('transcript_id', '')
            if not tx_id:
                continue

            r = raw[tx_id]
            if not r['gene_id']:
                r['gene_id']         = attrs.get('gene_id', '')
                # Ensembl uses gene_name; fall back to gene_id (locus name)
                r['gene_name']       = attrs.get('gene_name', '') or attrs.get('gene_id', '')
                r['transcript_name'] = attrs.get('transcript_name', tx_id)
                r['biotype']         = (attrs.get('transcript_biotype', '')
                                        or attrs.get('gene_biotype', ''))
                r['chrom']           = chrom
                r['strand']          = strand

            seg_len = end - start + 1
            if feature == 'five_prime_utr':
                r['utr5'].append(seg_len)
            elif feature == 'three_prime_utr':
                r['utr3'].append(seg_len)
            elif feature == 'CDS':
                r['cds'].append(seg_len)

    log.info(f"GTF parsed — {len(raw):,} transcripts annotated")

    # Store both versioned (T19E7.2.1) and base (T19E7.2) keys
    tx_map = {}
    for tx_id, r in raw.items():
        record = {
            'gene_id'        : r['gene_id'],
            'gene_name'      : r['gene_name'],
            'transcript_name': r['transcript_name'],
            'biotype'        : r['biotype'],
            'chrom'          : r['chrom'],
            'strand'         : r['strand'],
            'utr5_len'       : sum(r['utr5']),
            'cds_len'        : sum(r['cds']),
            'utr3_len'       : sum(r['utr3']),
        }
        tx_map[tx_id] = record
        # C. elegans transcript IDs don't have a simple dot-version suffix
        # like ENST IDs, but we store a no-last-segment variant anyway
        # e.g. "T19E7.2.1" → also store "T19E7.2" for FASTA header match
        parts = tx_id.rsplit('.', 1)
        if len(parts) == 2 and parts[1].isdigit():
            base = parts[0]
            if base not in tx_map:
                tx_map[base] = record
    return tx_map


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — FASTA header parser (Ensembl metazoa space-delimited format)
# ══════════════════════════════════════════════════════════════════════════════

def parse_ensembl_fasta_header(header: str) -> tuple:
    """
    Parse Ensembl metazoa cDNA FASTA header.

    Format:
      >T19E7.2.1 gene:WBGene00020878 gene_biotype:protein_coding
       transcript_biotype:protein_coding gene_symbol:T19E7.2
       description:hypothetical protein [Source:UniProtKB/TrEMBL;Acc:...]

    Fields extracted:
      tx_id        — e.g. "T19E7.2.1"
      gene_id      — WBGene ID (no version) e.g. "WBGene00020878"
      gene_name    — common name e.g. "unc-22", falls back to locus "T19E7.2"
      biotype      — e.g. "protein_coding"

    Returns: (tx_id, tx_id_base, gene_id, gene_name, gene_name, biotype)
    (tx_id_base = tx_id without trailing .digit suffix, for GTF lookup)
    """
    raw = header.lstrip('>').rstrip()
    tokens = raw.split()
    tx_id = tokens[0]

    # Build a key→value dict from remaining space-delimited tokens
    kv = {}
    for tok in tokens[1:]:
        if ':' in tok:
            k, _, v = tok.partition(':')
            kv[k] = v

    gene_id   = kv.get('gene', '').split('.')[0]          # strip version if any
    gene_name = kv.get('gene_symbol', '') or gene_id      # WormBase locus name
    biotype   = kv.get('transcript_biotype', '') or kv.get('gene_biotype', '')

    # Derive base ID: "T19E7.2.1" → "T19E7.2"  (strip trailing .digit)
    m = re.match(r'^(.+)\.\d+$', tx_id)
    tx_id_base = m.group(1) if m else tx_id

    return tx_id, tx_id_base, gene_id, gene_name, tx_id, biotype
    # Note: transcript_name field = tx_id (Ensembl metazoa doesn't provide
    # a separate display transcript name in the FASTA header; GTF has it)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Motif scanning (identical logic to human pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def scan_region(seq: str, pattern: re.Pattern, motif_type: str,
                region_name: str, cdna_offset: int) -> list:
    hits = []
    region_len = len(seq)
    if region_len == 0:
        return hits
    for m in pattern.finditer(seq):
        matched = m.group()
        if motif_type == 'PIS':
            tracts = re.findall(r'C+', matched)
            splits = re.split(r'C+', matched)
        else:
            tracts = re.findall(r'G+', matched)
            splits = re.split(r'G+', matched)
        loops = [s for s in splits if s]
        hits.append({
            'motif_type'    : motif_type,
            'region'        : region_name,
            'cdna_start'    : cdna_offset + m.start() + 1,
            'cdna_end'      : cdna_offset + m.end(),
            'local_start'   : m.start() + 1,
            'local_end'     : m.end(),
            'motif_length'  : len(matched),
            'sequence'      : matched,
            'tract_lengths' : ';'.join(str(len(t)) for t in tracts),
            'loop_lengths'  : ';'.join(str(len(l)) for l in loops),
            'region_length' : region_len,
            'density_per_kb': round(1000.0 / region_len, 6),
        })
    return hits


def scan_transcript(seq: str, tx_info: dict) -> tuple:
    tx_len  = len(seq)

    # =======================
    # Nucleotide composition
    # =======================
    g_count = seq.count("G")
    c_count = seq.count("C")
    a_count = seq.count("A")
    t_count = seq.count("T")

    g_percentage  = round(g_count * 100 / tx_len, 2) if tx_len else 0
    c_percentage  = round(c_count * 100 / tx_len, 2) if tx_len else 0
    gc_percentage = round((g_count + c_count) * 100 / tx_len, 2) if tx_len else 0

    u5_len  = tx_info.get('utr5_len', 0)
    cds_len = tx_info.get('cds_len',  0)
    u3_len  = tx_info.get('utr3_len', 0)

    annotated_total = u5_len + cds_len + u3_len
    if annotated_total > tx_len:
        scale  = tx_len / annotated_total
        u5_len  = int(u5_len  * scale)
        cds_len = int(cds_len * scale)
        u3_len  = tx_len - u5_len - cds_len

    u5_seq  = seq[:u5_len]
    cds_seq = seq[u5_len: u5_len + cds_len]
    u3_seq  = seq[u5_len + cds_len:]
    u3_len  = len(u3_seq)

    regions = [
        ("5'UTR", u5_seq,  0),
        ("CDS",   cds_seq, u5_len),
        ("3'UTR", u3_seq,  u5_len + cds_len),
    ]

    all_hits = []
    hit_map = {
        'PIS': {"5'UTR": [], 'CDS': [], "3'UTR": []},
        'PGS': {"5'UTR": [], 'CDS': [], "3'UTR": []},
    }
    for rname, rseq, offset in regions:
        if not rseq:
            continue
        for pat, mtype in [(PIS_PATTERN, 'PIS'), (PGS_PATTERN, 'PGS')]:
            h = scan_region(rseq, pat, mtype, rname, offset)
            hit_map[mtype][rname].extend(h)
            all_hits.extend(h)

    def loc_str(hits): return '|'.join(f"{h['cdna_start']}:{h['cdna_end']}" for h in hits)
    def seq_str(hits): return '|'.join(h['sequence'] for h in hits)
    def density(count, rlen): return round(count / rlen * 1000.0, 4) if rlen > 0 else 0.0

    row = {
        'wormbase_gene_id' : tx_info.get('gene_id', ''),
        'gene_symbol'      : tx_info.get('gene_name', ''),
        'gene_function'    : tx_info.get('gene_function', ''),
        'protein_domains'  : tx_info.get('protein_domains', ''),
        'transcript_id'    : tx_info.get('transcript_id', ''),
        'transcript_name'  : tx_info.get('transcript_name', ''),
        'biotype'          : tx_info.get('biotype', ''),
        'chromosome'       : tx_info.get('chrom', ''),
        'strand'           : tx_info.get('strand', ''),
        'transcript_length': tx_len,
        'g_count'          : g_count,
        'c_count'          : c_count,
        'a_count'          : a_count,
        't_count'          : t_count,
        'g_percentage'     : g_percentage,
        'c_percentage'     : c_percentage,
        'gc_percentage'    : gc_percentage,
        'utr5_length'      : u5_len,
        'cds_length'       : cds_len,
        'utr3_length'      : u3_len,
        # PIS
        'pis_utr5_count'             : len(hit_map['PIS']["5'UTR"]),
        'pis_cds_count'              : len(hit_map['PIS']['CDS']),
        'pis_utr3_count'             : len(hit_map['PIS']["3'UTR"]),
        'pis_total'                  : sum(len(v) for v in hit_map['PIS'].values()),
        'pis_utr5_density_per_kb'    : density(len(hit_map['PIS']["5'UTR"]),  u5_len),
        'pis_cds_density_per_kb'     : density(len(hit_map['PIS']['CDS']),    cds_len),
        'pis_utr3_density_per_kb'    : density(len(hit_map['PIS']["3'UTR"]), u3_len),
        'pis_total_density_per_kb'   : density(sum(len(v) for v in hit_map['PIS'].values()), tx_len),
        'pis_utr5_locations'         : loc_str(hit_map['PIS']["5'UTR"]),
        'pis_cds_locations'          : loc_str(hit_map['PIS']['CDS']),
        'pis_utr3_locations'         : loc_str(hit_map['PIS']["3'UTR"]),
        'pis_utr5_sequences'         : seq_str(hit_map['PIS']["5'UTR"]),
        'pis_cds_sequences'          : seq_str(hit_map['PIS']['CDS']),
        'pis_utr3_sequences'         : seq_str(hit_map['PIS']["3'UTR"]),
        # PGS
        'pgs_utr5_count'             : len(hit_map['PGS']["5'UTR"]),
        'pgs_cds_count'              : len(hit_map['PGS']['CDS']),
        'pgs_utr3_count'             : len(hit_map['PGS']["3'UTR"]),
        'pgs_total'                  : sum(len(v) for v in hit_map['PGS'].values()),
        'pgs_utr5_density_per_kb'    : density(len(hit_map['PGS']["5'UTR"]),  u5_len),
        'pgs_cds_density_per_kb'     : density(len(hit_map['PGS']['CDS']),    cds_len),
        'pgs_utr3_density_per_kb'    : density(len(hit_map['PGS']["3'UTR"]), u3_len),
        'pgs_total_density_per_kb'   : density(sum(len(v) for v in hit_map['PGS'].values()), tx_len),
        'pgs_utr5_locations'         : loc_str(hit_map['PGS']["5'UTR"]),
        'pgs_cds_locations'          : loc_str(hit_map['PGS']['CDS']),
        'pgs_utr3_locations'         : loc_str(hit_map['PGS']["3'UTR"]),
        'pgs_utr5_sequences'         : seq_str(hit_map['PGS']["5'UTR"]),
        'pgs_cds_sequences'          : seq_str(hit_map['PGS']['CDS']),
        'pgs_utr3_sequences'         : seq_str(hit_map['PGS']["3'UTR"]),
    }
    return all_hits, row


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — FASTA streamer (identical to human pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def stream_fasta(fasta_path: Path):
    opener = gzip.open if str(fasta_path).endswith('.gz') else open
    header = None
    buf = []
    with opener(fasta_path, 'rt', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(buf)
                header = line
                buf = []
            else:
                buf.append(line.upper().replace('U', 'T'))
    if header is not None:
        yield header, ''.join(buf)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Gene-level summary
# ══════════════════════════════════════════════════════════════════════════════

def build_summary(catalogue_rows: list) -> list:
    by_gene = defaultdict(list)
    for row in catalogue_rows:
        by_gene[row['wormbase_gene_id']].append(row)

    summary = []
    for gene_id, rows in by_gene.items():
        n  = len(rows)
        r0 = rows[0]

        def pct(c):    return round(c / n * 100, 2) if n else 0.0
        def avg(vals): return round(sum(vals) / len(vals), 4) if vals else 0.0

        def _any(p):   return sum(1 for r in rows if r[f'{p}_total'] > 0)
        def _reg(p, g): return sum(1 for r in rows if r[f'{p}_{g}_count'] > 0)
        def _ad(p, g): return avg([r[f'{p}_{g}_density_per_kb'] for r in rows])
        def _atd(p):   return avg([r[f'{p}_total_density_per_kb'] for r in rows])

        summary.append({
            'wormbase_gene_id': gene_id,
            'gene_symbol'     : r0['gene_symbol'],
            'gene_function'   : r0.get('gene_function', ''),
            'protein_domains' : r0.get('protein_domains', ''),
            'chromosome'      : r0['chromosome'],
            'strand'          : r0['strand'],
            'biotype'         : r0['biotype'],
            'n_transcripts'   : n,
            # PIS
            'pis_total_hits'               : sum(r['pis_total'] for r in rows),
            'pis_transcripts_with_any'     : _any('pis'),
            'pis_transcripts_with_utr5'    : _reg('pis', 'utr5'),
            'pis_transcripts_with_cds'     : _reg('pis', 'cds'),
            'pis_transcripts_with_utr3'    : _reg('pis', 'utr3'),
            'pis_pct_transcripts_with_any' : pct(_any('pis')),
            'pis_pct_transcripts_with_utr5': pct(_reg('pis', 'utr5')),
            'pis_pct_transcripts_with_cds' : pct(_reg('pis', 'cds')),
            'pis_pct_transcripts_with_utr3': pct(_reg('pis', 'utr3')),
            'pis_avg_density_utr5'         : _ad('pis', 'utr5'),
            'pis_avg_density_cds'          : _ad('pis', 'cds'),
            'pis_avg_density_utr3'         : _ad('pis', 'utr3'),
            'pis_avg_total_density'        : _atd('pis'),
            # PGS
            'pgs_total_hits'               : sum(r['pgs_total'] for r in rows),
            'pgs_transcripts_with_any'     : _any('pgs'),
            'pgs_transcripts_with_utr5'    : _reg('pgs', 'utr5'),
            'pgs_transcripts_with_cds'     : _reg('pgs', 'cds'),
            'pgs_transcripts_with_utr3'    : _reg('pgs', 'utr3'),
            'pgs_pct_transcripts_with_any' : pct(_any('pgs')),
            'pgs_pct_transcripts_with_utr5': pct(_reg('pgs', 'utr5')),
            'pgs_pct_transcripts_with_cds' : pct(_reg('pgs', 'cds')),
            'pgs_pct_transcripts_with_utr3': pct(_reg('pgs', 'utr3')),
            'pgs_avg_density_utr5'         : _ad('pgs', 'utr5'),
            'pgs_avg_density_cds'          : _ad('pgs', 'cds'),
            'pgs_avg_density_utr3'         : _ad('pgs', 'utr3'),
            'pgs_avg_total_density'        : _atd('pgs'),
        })

    summary.sort(key=lambda r: r['pgs_total_hits'] + r['pis_total_hits'], reverse=True)
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Staged CSV writer (writes to /tmp first, then copies to Windows)
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(path: Path, fieldnames: list, rows: list, label: str,
              tmp_dir: Path):
    log.info(f"Writing {label} → {path.name}  ({len(rows):,} rows)")
    tmp_path = tmp_dir / path.name

    with open(tmp_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    size_mb = tmp_path.stat().st_size / 1024 / 1024
    log.info(f"  staged → {tmp_path}  ({size_mb:.1f} MB)")

    for attempt in range(1, 6):
        try:
            if path.exists():
                path.unlink()
            shutil.copy2(tmp_path, path)
            log.info(f"  ✓ {path.name}  ({path.stat().st_size / 1024 / 1024:.1f} MB)")
            tmp_path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt < 5:
                wait = 15 * attempt
                log.warning(f"  PermissionError (attempt {attempt}/5) — "
                            f"OneDrive lock. Retrying in {wait}s …")
                time.sleep(wait)
            else:
                log.error(f"  FAILED to copy to {path}. File preserved at: {tmp_path}")
                raise


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='C. elegans WBcel235 PIS/PGS transcriptome scanner'
    )

    parser.add_argument('--fasta', required=True)
    parser.add_argument('--gtf', required=True)
    parser.add_argument('--functional_descriptions', required=False, default=None, help='WormBase functional descriptions')

    parser.add_argument('--protein_domains', required=False, default=None, help='WormBase protein domains')
    parser.add_argument('--outdir', required=True)

    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Limit to N transcripts (0=all; use for testing)'
    )

    args = parser.parse_args()

    fasta_path = Path(args.fasta)
    gtf_path = Path(args.gtf)
    outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    for p in (fasta_path, gtf_path):
        if not p.exists():
            log.error(f"File not found: {p}")
            sys.exit(1)

    # ── 1. Parse GTF ──────────────────────────────────────────────────────────
    tx_map = parse_gtf(gtf_path)

    gene_descriptions = {}
    protein_domains = {}

    if args.functional_descriptions:
        log.info("Loading functional descriptions...")
        gene_descriptions = load_functional_descriptions(
            args.functional_descriptions
        )

    if args.protein_domains:
        log.info("Loading protein domains...")
        protein_domains = load_protein_domains(
            args.protein_domains
        )

    # ── 2. Diagnostic: inspect first 3 headers ────────────────────────────────
    log.info("--- Header format diagnostic (first 3 transcripts) ---")
    n_diag = 0
    for header, seq in stream_fasta(fasta_path):
        tx_id, tx_id_base, fasta_gene_id, fasta_gene_name, _, fasta_biotype = \
            parse_ensembl_fasta_header(header)
        gtf_hit = bool(tx_map.get(tx_id) or tx_map.get(tx_id_base))
        log.info(f"  HEADER : {header[:120]}")
        log.info(f"  PARSED : tx={tx_id}  base={tx_id_base}  gene_id={fasta_gene_id}  "
                 f"gene_name={fasta_gene_name}  biotype={fasta_biotype}  GTF_hit={gtf_hit}")
        n_diag += 1
        if n_diag >= 3:
            break
    log.info("--- End diagnostic ---")

    # ── 3. Scan all transcripts ───────────────────────────────────────────────
    all_hits_rows  = []
    catalogue_rows = []
    n_tx      = 0
    n_gtf_hit = 0
    n_with_pis = 0
    n_with_pgs = 0

    for header, seq in tqdm(stream_fasta(fasta_path), desc="Transcripts", unit="tx"):
        tx_id, tx_id_base, fasta_gene_id, fasta_gene_name, _, fasta_biotype = \
            parse_ensembl_fasta_header(header)

        info = tx_map.get(tx_id) or tx_map.get(tx_id_base) or {}
        if info:
            n_gtf_hit += 1

        tx_info = {
            'transcript_id'  : tx_id,
            'gene_id'        : info.get('gene_id')         or fasta_gene_id   or tx_id_base,
            'gene_name'      : info.get('gene_name')       or fasta_gene_name or '',
            'transcript_name': info.get('transcript_name') or tx_id,
            'biotype'        : info.get('biotype')         or fasta_biotype   or '',
            'chrom'          : info.get('chrom',  ''),
            'strand'         : info.get('strand', ''),
            'utr5_len'       : info.get('utr5_len', 0),
            'cds_len'        : info.get('cds_len',  0),
            'utr3_len'       : info.get('utr3_len', 0),
        }
        gid = tx_info['gene_id']

        tx_info['gene_function'] = \
            gene_descriptions.get(gid, '')

        tx_info['protein_domains'] = \
            protein_domains.get(gid, '')

        tx_hits, cat_row = scan_transcript(seq, tx_info)

        for h in tx_hits:
            h['wormbase_gene_id']  = tx_info['gene_id']
            h['gene_symbol']       = tx_info['gene_name']
            h['gene_function'] = tx_info['gene_function']
            h['protein_domains'] = tx_info['protein_domains']
            h['transcript_id']     = tx_info['transcript_id']
            h['transcript_name']   = tx_info['transcript_name']
            h['biotype']           = tx_info['biotype']
            h['chromosome']        = tx_info['chrom']
            h['strand']            = tx_info['strand']
            h['transcript_length'] = len(seq)
            h['utr5_length']       = tx_info['utr5_len']
            h['cds_length']        = tx_info['cds_len']
            h['utr3_length']       = tx_info['utr3_len']

        all_hits_rows.extend(tx_hits)
        catalogue_rows.append(cat_row)

        if cat_row['pis_total'] > 0: n_with_pis += 1
        if cat_row['pgs_total'] > 0: n_with_pgs += 1

        n_tx += 1
        if args.limit and n_tx >= args.limit:
            log.info(f"--limit {args.limit} reached.")
            break

    log.info(f"Scan complete: {n_tx:,} transcripts, "
             f"{len(all_hits_rows):,} total hits")
    log.info(f"  GTF matched: {n_gtf_hit:,}/{n_tx:,} "
             f"({n_gtf_hit/n_tx*100:.1f}%)")
    log.info(f"  Transcripts with PIS: {n_with_pis:,} ({n_with_pis/n_tx*100:.1f}%)")
    log.info(f"  Transcripts with PGS: {n_with_pgs:,} ({n_with_pgs/n_tx*100:.1f}%)")

    # ── 4. Split by motif type ────────────────────────────────────────────────
    pis_hits = [h for h in all_hits_rows if h['motif_type'] == 'PIS']
    pgs_hits = [h for h in all_hits_rows if h['motif_type'] == 'PGS']
    log.info(f"  PIS hits: {len(pis_hits):,}   PGS hits: {len(pgs_hits):,}")

    # ── 5. Build gene summary ─────────────────────────────────────────────────
    log.info("Building gene-level summary …")
    summary_rows = build_summary(catalogue_rows)
    log.info(f"  Genes: {len(summary_rows):,}")

    # ── 6. Write CSVs (staged via /tmp) ───────────────────────────────────────
    tmp_stage = Path(tempfile.gettempdir()) / f'pis_pgs_{SPECIES_TAG}'
    tmp_stage.mkdir(exist_ok=True)
    log.info(f"Staging dir: {tmp_stage}")

    tag = SPECIES_TAG
    write_csv(outdir / f'PGS_Hits_{tag}.csv',       HITS_FIELDNAMES,       pgs_hits,        'PGS Hits',      tmp_stage)
    write_csv(outdir / f'PGS_Catalogue_{tag}.csv',  CATALOGUE_FIELDNAMES,  catalogue_rows,  'PGS Catalogue', tmp_stage)
    write_csv(outdir / f'PGS_Summary_{tag}.csv',    SUMMARY_FIELDNAMES,    summary_rows,    'PGS Summary',   tmp_stage)
    write_csv(outdir / f'PIS_Hits_{tag}.csv',       HITS_FIELDNAMES,       pis_hits,        'PIS Hits',      tmp_stage)
    write_csv(outdir / f'PIS_Catalogue_{tag}.csv',  CATALOGUE_FIELDNAMES,  catalogue_rows,  'PIS Catalogue', tmp_stage)
    write_csv(outdir / f'PIS_Summary_{tag}.csv',    SUMMARY_FIELDNAMES,    summary_rows,    'PIS Summary',   tmp_stage)

    # ── 7. Write stats JSON ───────────────────────────────────────────────────
    from statistics import mean, median   # near imports

    # ============================================================================
    # Density summaries
    # ============================================================================

    def density_stats(rows, density_col, count_col):
        """
        Returns:
            mean_all
            median_all
            mean_positive
            median_positive
        """
        # Includes every transcript
        all_density = [
            r[density_col]
            for r in rows
        ]

        # Includes only transcripts containing >= 1 motif
        positive_density = [
            r[density_col]
            for r in rows
            if r[count_col] > 0
        ]

        return {
            "mean_all": round(mean(all_density), 4) if all_density else 0,
            "median_all": round(median(all_density), 4) if all_density else 0,
            "mean_positive": round(mean(positive_density), 4) if positive_density else 0,
            "median_positive": round(median(positive_density), 4) if positive_density else 0,
        }

    pis_utr5 = density_stats(catalogue_rows, "pis_utr5_density_per_kb", "pis_utr5_count")
    pis_cds  = density_stats(catalogue_rows, "pis_cds_density_per_kb", "pis_cds_count")
    pis_utr3 = density_stats(catalogue_rows, "pis_utr3_density_per_kb", "pis_utr3_count")

    pgs_utr5 = density_stats(catalogue_rows, "pgs_utr5_density_per_kb", "pgs_utr5_count")
    pgs_cds  = density_stats(catalogue_rows, "pgs_cds_density_per_kb", "pgs_cds_count")
    pgs_utr3 = density_stats(catalogue_rows, "pgs_utr3_density_per_kb", "pgs_utr3_count")

    # =======================
    # Composition summaries
    # =======================
    gc_values = [r["gc_percentage"] for r in catalogue_rows]
    g_values  = [r["g_percentage"] for r in catalogue_rows]
    c_values  = [r["c_percentage"] for r in catalogue_rows]

    # ===== stats =====
    stats = {
        'species'                   : 'Caenorhabditis elegans',
        'assembly'                  : 'WBcel235',
        'ensembl_release'           : '113',
        'total_transcripts_scanned' : n_tx,
        'gtf_annotation_matched'    : n_gtf_hit,
        'gtf_match_pct'             : round(n_gtf_hit / n_tx * 100, 2) if n_tx else 0,
        
        # ---------------- Composition ----------------
        'gc_percentage_mean'        : round(mean(gc_values), 2) if gc_values else 0,
        'gc_percentage_median'      : round(median(gc_values), 2) if gc_values else 0,
        'g_percentage_mean'         : round(mean(g_values), 2) if g_values else 0,
        'g_percentage_median'       : round(median(g_values), 2) if g_values else 0,
        'c_percentage_mean'         : round(mean(c_values), 2) if c_values else 0,
        'c_percentage_median'       : round(median(c_values), 2) if c_values else 0,

        'total_pis_hits'            : len(pis_hits),
        'total_pgs_hits'            : len(pgs_hits),
        'transcripts_with_pis'      : n_with_pis,
        'transcripts_with_pgs'      : n_with_pgs,
        'pct_tx_with_pis'           : round(n_with_pis / n_tx * 100, 2) if n_tx else 0,
        'pct_tx_with_pgs'           : round(n_with_pgs / n_tx * 100, 2) if n_tx else 0,

        # ---------------- PIS ----------------
        'pis_utr5_density_mean_all_per_kb'       : pis_utr5["mean_all"],
        'pis_utr5_density_median_all_per_kb'     : pis_utr5["median_all"],
        'pis_utr5_density_mean_positive_per_kb'  : pis_utr5["mean_positive"],
        'pis_utr5_density_median_positive_per_kb': pis_utr5["median_positive"],

        'pis_cds_density_mean_all_per_kb'        : pis_cds["mean_all"],
        'pis_cds_density_median_all_per_kb'      : pis_cds["median_all"],
        'pis_cds_density_mean_positive_per_kb'   : pis_cds["mean_positive"],
        'pis_cds_density_median_positive_per_kb' : pis_cds["median_positive"],

        'pis_utr3_density_mean_all_per_kb'       : pis_utr3["mean_all"],
        'pis_utr3_density_median_all_per_kb'     : pis_utr3["median_all"],
        'pis_utr3_density_mean_positive_per_kb'  : pis_utr3["mean_positive"],
        'pis_utr3_density_median_positive_per_kb': pis_utr3["median_positive"],

        # ---------------- PGS ----------------
        'pgs_utr5_density_mean_all_per_kb'       : pgs_utr5["mean_all"],
        'pgs_utr5_density_median_all_per_kb'     : pgs_utr5["median_all"],
        'pgs_utr5_density_mean_positive_per_kb'  : pgs_utr5["mean_positive"],
        'pgs_utr5_density_median_positive_per_kb': pgs_utr5["median_positive"],

        'pgs_cds_density_mean_all_per_kb'        : pgs_cds["mean_all"],
        'pgs_cds_density_median_all_per_kb'      : pgs_cds["median_all"],
        'pgs_cds_density_mean_positive_per_kb'   : pgs_cds["mean_positive"],
        'pgs_cds_density_median_positive_per_kb' : pgs_cds["median_positive"],

        'pgs_utr3_density_mean_all_per_kb'       : pgs_utr3["mean_all"],
        'pgs_utr3_density_median_all_per_kb'     : pgs_utr3["median_all"],
        'pgs_utr3_density_mean_positive_per_kb'  : pgs_utr3["mean_positive"],
        'pgs_utr3_density_median_positive_per_kb': pgs_utr3["median_positive"],

        'pis_pattern'               : r'C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}',
        'pgs_pattern'               : r'G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}',
    }
    
    tmp_stats = tmp_stage / f'scan_stats_{tag}.json'
    stats_path = outdir / f'scan_stats_{tag}.json'
    with open(tmp_stats, 'w') as fh:
        json.dump(stats, fh, indent=2)
    for attempt in range(1, 6):
        try:
            shutil.copy2(tmp_stats, stats_path)
            tmp_stats.unlink(missing_ok=True)
            break
        except PermissionError:
            time.sleep(15 * attempt)
    log.info(f"Stats → {stats_path}")

    log.info("=" * 60)
    log.info("ALL DONE")
    for fname in [f'PGS_Hits_{tag}.csv', f'PGS_Catalogue_{tag}.csv', f'PGS_Summary_{tag}.csv',
                  f'PIS_Hits_{tag}.csv', f'PIS_Catalogue_{tag}.csv', f'PIS_Summary_{tag}.csv',
                  f'scan_stats_{tag}.json']:
        p = outdir / fname
        if p.exists():
            log.info(f"  ✓  {fname}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    log.info("=" * 60)
    log.info(f"\nNext step — generate HTML dashboard:")
    log.info(f'  python3 make_summary_html.py --datadir "{outdir}" --species celegans')


if __name__ == '__main__':
    main()