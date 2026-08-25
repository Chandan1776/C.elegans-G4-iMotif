"""
========================================================================
PIS & PGS Genome-Wide DNA Catalogue — C. elegans WBcel235 (Forward Only)
========================================================================
Performs a genome-wide forward-strand (+) scan of the C. elegans genome.
  - PIS  (i-Motif):        C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}
  - PGS  (G-Quadruplex):   G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}

Reference files (WBcel235 assembly):
  FASTA : Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz   (Ensembl)
  GFF3  : caenorhabditis_elegans.PRJNA13758.WBPS19.annotations.gff3.gz  (WormBase)
"""

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
from bisect import bisect_left
from statistics import mean, median

from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Motif patterns for genome-wide forward DNA scanning ───────────────────────
PIS_PATTERN = re.compile(r'C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}[ATCG]{1,7}C{3,5}')
PGS_PATTERN = re.compile(r'G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}[ATCG]{1,7}G{3,5}')

SPECIES_TAG = 'DNA_Celegans_WBcel235'

DIRECT_FEATURES = {
    'gene', 'mRNA', 'exon', 'CDS',
    'five_prime_UTR', 'three_prime_UTR',
    'intron', 'promoter', 'TSS_region',
    'lincRNA', 'miRNA', 'piRNA', 'ncRNA',
    'snoRNA', 'snRNA', 'tRNA', 'rRNA', 'antisense_RNA',
}

GFF3_OVERLAP_FEATURES = {
    'G_quartet',
    'TF_binding_site',
    'DNaseI_hypersensitive_site',
    'enhancer',
}

GFF3_OVERLAP_MAP = {
    'G_quartet': ('overlaps_G_quartet', 'g_quartet_id'),
    'TF_binding_site': ('overlaps_TF_binding_site', 'tf_binding_site_id'),
    'DNaseI_hypersensitive_site': ('overlaps_DNaseI_site', 'dnase_site_id'),
    'enhancer': ('overlaps_enhancer', 'enhancer_id'),
}

RNA_GENE_TYPES = {
    'lincRNA', 'miRNA', 'piRNA', 'ncRNA',
    'snoRNA', 'snRNA', 'tRNA', 'rRNA', 'antisense_RNA',
}

FINE_REGION_PRIORITY = ['five_prime_UTR', 'CDS', 'three_prime_UTR', 'intron']
BROAD_REGION_PRIORITY = ['promoter', 'TSS_region', 'gene_body', 'intergenic']

# ── Column definitions ────────────────────────────────────────────────────────
HITS_FIELDNAMES = [
    'chromosome', 'start', 'end', 'motif_type', 'motif_sequence',
    'motif_length', 'tract_lengths', 'loop_lengths',
    'wormbase_gene_id', 'gene_symbol', 'gene_function', 'protein_domains', 'transcript_id', 'biotype', 'gene_strand',
    'broad_region', 'fine_region',
    'overlaps_G_quartet', 'g_quartet_id',
    'overlaps_TF_binding_site', 'tf_binding_site_id',
    'overlaps_DNaseI_site', 'dnase_site_id',
    'overlaps_enhancer', 'enhancer_id',
]

CATALOGUE_FIELDNAMES = [
    'wormbase_gene_id', 'gene_symbol', 'gene_function', 'protein_domains', 'chromosome', 'strand', 'biotype',
    'gene_start', 'gene_end', 'gene_length',
    'g_count', 'c_count', 'a_count', 't_count',
    'g_percentage', 'c_percentage', 'gc_percentage',
    'five_prime_UTR_length', 'CDS_length', 'intron_length', 'three_prime_UTR_length', 'exon_length',
    'promoter_length', 'TSS_region_length', 'gene_body_length', 'intergenic_length',
    'pis_five_prime_UTR_count', 'pis_CDS_count', 'pis_intron_count', 'pis_three_prime_UTR_count', 'pis_exon_count',
    'pis_promoter_count', 'pis_TSS_region_count', 'pis_gene_body_count', 'pis_intergenic_count', 'pis_total',
    'pis_five_prime_UTR_density_per_kb', 'pis_CDS_density_per_kb', 'pis_intron_density_per_kb', 'pis_three_prime_UTR_density_per_kb',
    'pis_exon_density_per_kb', 'pis_promoter_density_per_kb', 'pis_TSS_region_density_per_kb', 'pis_gene_body_density_per_kb',
    'pis_intergenic_density_per_kb', 'pis_total_density_per_kb',
    'pgs_five_prime_UTR_count', 'pgs_CDS_count', 'pgs_intron_count', 'pgs_three_prime_UTR_count', 'pgs_exon_count',
    'pgs_promoter_count', 'pgs_TSS_region_count', 'pgs_gene_body_count', 'pgs_intergenic_count', 'pgs_total',
    'pgs_five_prime_UTR_density_per_kb', 'pgs_CDS_density_per_kb', 'pgs_intron_density_per_kb', 'pgs_three_prime_UTR_density_per_kb',
    'pgs_exon_density_per_kb', 'pgs_promoter_density_per_kb', 'pgs_TSS_region_density_per_kb', 'pgs_gene_body_density_per_kb',
    'pgs_intergenic_density_per_kb', 'pgs_total_density_per_kb',
]

SUMMARY_FIELDNAMES = [
    'wormbase_gene_id', 'gene_symbol','gene_function', 'protein_domains', 'chromosome', 'strand', 'biotype', 'gene_length',
    'pis_total_hits', 'pgs_total_hits', 'pis_total_density_per_kb', 'pgs_total_density_per_kb',
    'pis_CDS_hits', 'pgs_CDS_hits', 'pis_five_prime_UTR_hits', 'pgs_five_prime_UTR_hits',
    'pis_three_prime_UTR_hits', 'pgs_three_prime_UTR_hits', 'pis_intron_hits', 'pgs_intron_hits', 'pis_promoter_hits', 'pgs_promoter_hits',
]

# ── Parsing External Annotations ──────────────────────────────────────────────
def load_functional_descriptions(desc_file):
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
            concise_lines, automated_lines = [], []
            j = i + 1
            mode = None
            while j < len(lines):
                current = lines[j].rstrip()
                if current.strip() == "=":
                    break
                if current.startswith("Concise description:"):
                    mode = "concise"
                    concise_lines.append(current.replace("Concise description:", "").strip())
                elif current.startswith("Automated description:"):
                    mode = "automated"
                    automated_lines.append(current.replace("Automated description:", "").strip())
                else:
                    if mode == "concise":
                        concise_lines.append(current.strip())
                    elif mode == "automated":
                        automated_lines.append(current.strip())
                j += 1
            concise_text = " ".join(concise_lines).strip()
            automated_text = " ".join(automated_lines).strip()
            descriptions[gene_id] = concise_text if concise_text else automated_text
            i = j 
        i += 1
    return descriptions

def load_protein_domains(domain_file):
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
                        domain_names.append(item.split('"')[1])
                    except Exception:
                        pass
            if gene_id not in domains:
                domains[gene_id] = set()
            domains[gene_id].update(domain_names)
    return {gid: "; ".join(sorted(vals)) for gid, vals in domains.items()}

# ── Annotation Manager ────────────────────────────────────────────────────────
class GenomeAnnotation:
    def __init__(self):
        self.features = defaultdict(lambda: defaultdict(list))
        self.genes = {}
        self.tx2gene = {}
        self._index = {}

    def _add(self, chrom, ftype, start0, end0, meta):
        self.features[chrom][ftype].append((start0, end0, meta))

    def sort_all(self):
        for chrom, ftypes in self.features.items():
            for ftype, ivals in ftypes.items():
                ivals.sort(key=lambda x: x[0])
                starts = [iv[0] for iv in ivals]
                prefix_max_ends = []
                max_end = 0
                for _, end0, _ in ivals:
                    max_end = max(max_end, end0)
                    prefix_max_ends.append(max_end)
                self._index[(chrom, ftype)] = (starts, prefix_max_ends)
        log.info("  Annotation intervals sorted.")

    def overlapping(self, chrom, ftype, pos0, end0):
        ivals = self.features.get(chrom, {}).get(ftype, [])
        if not ivals:
            return []
        starts, prefix_max_ends = self._index.get((chrom, ftype), ([], []))
        hi = bisect_left(starts, end0)
        results = []
        for i in range(hi - 1, -1, -1):
            if prefix_max_ends[i] <= pos0:
                break
            iv = ivals[i]
            if iv[1] > pos0:
                results.append(iv)
        return results

def _gff3_attrs(attr_str: str) -> dict:
    attrs = {}
    for part in attr_str.split(';'):
        part = part.strip()
        if '=' in part:
            k, _, v = part.partition('=')
            attrs[k.strip()] = v.strip()
    return attrs

def _strip_gff3_prefix(value: str) -> str:
    value = (value or '').strip()
    for prefix in ('Gene:', 'gene:', 'Transcript:', 'transcript:', 'CDS:', 'cds:'):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value

def _first_id(value: str) -> str:
    if not value:
        return ''
    return _strip_gff3_prefix(value.split(',')[0])

def _wormbase_gene_symbol(attrs: dict, gene_id: str) -> str:
    if attrs.get('locus'):
        return attrs['locus']
    if attrs.get('Alias'):
        first_alias = attrs['Alias'].split(',')[0].strip()
        if first_alias:
            return first_alias
    if attrs.get('sequence_name'):
        return attrs['sequence_name']
    name = _strip_gff3_prefix(attrs.get('Name', ''))
    return name or gene_id

def parse_gff3(gff3_path: Path) -> GenomeAnnotation:
    log.info(f"Parsing GFF3: {gff3_path}")
    ann = GenomeAnnotation()
    opener = gzip.open if str(gff3_path).endswith('.gz') else open
    n_lines, n_feat = 0, 0

    with opener(gff3_path, 'rt', errors='replace') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            n_lines += 1
            if n_lines % 2_000_000 == 0:
                log.info(f"  … {n_lines:,} GFF3 lines, {n_feat:,} features loaded")

            cols = line.rstrip('\n').split('\t')
            if len(cols) < 9:
                continue

            chrom, source, ftype = cols[0], cols[1], cols[2]
            start0, end0 = int(cols[3]) - 1, int(cols[4])

            if ftype not in DIRECT_FEATURES and ftype not in GFF3_OVERLAP_FEATURES:
                continue

            attrs = _gff3_attrs(attr_str=cols[8])
            feat_id, parent, biotype = attrs.get('ID', ''), attrs.get('Parent', ''), attrs.get('biotype', '')

            if ftype == 'gene' and source == 'WormBase':
                gene_id = _first_id(feat_id)
                gene_symbol = _wormbase_gene_symbol(attrs, gene_id)
                ann.genes[gene_id] = {
                    'chrom': chrom, 'strand': cols[6], 'start0': start0, 'end0': end0, 'gene_symbol': gene_symbol, 'biotype': biotype,
                }
                ann._add(chrom, 'gene', start0, end0, {
                    'wormbase_gene_id': gene_id, 'gene_symbol': gene_symbol, 'biotype': biotype, 'strand': cols[6],
                })
                n_feat += 1
            elif ftype == 'mRNA' or ftype in RNA_GENE_TYPES:
                tx_id = _first_id(feat_id)
                par_gene = _first_id(parent)
                if tx_id:
                    ann.tx2gene[tx_id] = par_gene
                ann._add(chrom, ftype, start0, end0, {
                    'transcript_id': tx_id, 'wormbase_gene_id': par_gene, 'biotype': biotype or ftype, 'strand': cols[6],
                })
                n_feat += 1
            elif ftype in ('exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR', 'intron', 'promoter', 'TSS_region'):
                par_tx = _first_id(parent)
                par_gene = par_tx if parent.startswith(('Gene:', 'gene:')) else ann.tx2gene.get(par_tx, '')
                tx_id = '' if parent.startswith(('Gene:', 'gene:')) else par_tx
                ann._add(chrom, ftype, start0, end0, {
                    'transcript_id': tx_id, 'wormbase_gene_id': par_gene, 'parent_tx': tx_id, 'strand': cols[6],
                })
                n_feat += 1
            elif ftype in GFF3_OVERLAP_FEATURES:
                feature_id = _feature_id_from_attrs(attrs, f'{ftype}:{chrom}:{cols[3]}-{cols[4]}')
                ann._add(chrom, ftype, start0, end0, {'feature_id': feature_id, 'feature_type': ftype, 'strand': cols[6]})
                n_feat += 1

    for chrom, ftypes in ann.features.items():
        for ftype in ('exon', 'CDS', 'five_prime_UTR', 'three_prime_UTR', 'intron'):
            for _, _, meta in ftypes.get(ftype, []):
                if not meta.get('wormbase_gene_id') and meta.get('parent_tx'):
                    meta['wormbase_gene_id'] = ann.tx2gene.get(meta['parent_tx'], '')

    ann.sort_all()
    log.info(f"GFF3 parsed — {len(ann.genes):,} genes, {n_feat:,} features total")
    return ann

# ── Fasta Reader & Streamer ───────────────────────────────────────────────────
def stream_genome_fasta(fasta_path: Path):
    opener = gzip.open if str(fasta_path).endswith('.gz') else open
    chrom = None
    buf = []
    with opener(fasta_path, 'rt', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\n')
            if line.startswith('>'):
                if chrom is not None:
                    yield chrom, ''.join(buf)
                chrom = line.lstrip('>').split()[0]
                buf = []
            else:
                buf.append(line.upper().replace('U', 'T'))
    if chrom is not None:
        yield chrom, ''.join(buf)

# ── Scanning Core (Forward Only) ──────────────────────────────────────────────
def scan_sequence(seq: str, pattern: re.Pattern, motif_type: str, chrom: str, chrom_offset: int = 0) -> list:
    hits = []
    for m in pattern.finditer(seq):
        matched = m.group()
        if motif_type == 'PIS':
            tracts = re.findall(r'C+', matched)
            splits = re.split(r'C+', matched)
        else:
            tracts = re.findall(r'G+', matched)
            splits = re.split(r'G+', matched)
        loops = [s for s in splits if s]

        abs_start1 = chrom_offset + m.start() + 1
        abs_end1 = chrom_offset + m.end()

        hits.append({
            'chromosome': chrom, 'start': abs_start1, 'end': abs_end1, 'motif_sequence': matched, 'motif_type': motif_type,
            'motif_length': len(matched), 'tract_lengths': ';'.join(str(len(t)) for t in tracts), 'loop_lengths': ';'.join(str(len(l)) for l in loops),
            'wormbase_gene_id': '', 'gene_symbol': '', 'gene_function': '', 'protein_domains': '', 'transcript_id': '', 'biotype': '', 'gene_strand': '',
            'broad_region': '', 'fine_region': '',
            'overlaps_G_quartet': False, 'g_quartet_id': '', 'overlaps_TF_binding_site': False, 'tf_binding_site_id': '',
            'overlaps_DNaseI_site': False, 'dnase_site_id': '', 'overlaps_enhancer': False, 'enhancer_id': '',
        })
    return hits

def scan_chromosome(chrom: str, seq: str) -> list:
    hits = []
    for pat, mtype in [(PGS_PATTERN, 'PGS'), (PIS_PATTERN, 'PIS')]:
        hits.extend(scan_sequence(seq, pat, mtype, chrom))
    return hits

# ── Hit Annotation Logic ──────────────────────────────────────────────────────
def annotate_hit(hit: dict, ann: GenomeAnnotation) -> dict:
    chrom = hit['chromosome']
    start0, end0 = hit['start'] - 1, hit['end']

    final_region = ''
    region_iv = None
    for ftype in ('promoter', 'TSS_region') + tuple(FINE_REGION_PRIORITY):
        hits = ann.overlapping(chrom, ftype, start0, end0)
        if hits:
            final_region = ftype
            region_iv = hits[0]
            break

    gene_overlaps = ann.overlapping(chrom, 'gene', start0, end0)
    if gene_overlaps:
        g_meta = gene_overlaps[0][2]
        gene_id = g_meta.get('wormbase_gene_id', '')
        gene_symbol = g_meta.get('gene_symbol', gene_id)
        biotype = g_meta.get('biotype', '')
        annotated_strand = g_meta.get('strand', '')
    else:
        region_gene_id = region_iv[2].get('wormbase_gene_id', '') if region_iv else ''
        g_info = ann.genes.get(region_gene_id, {})
        gene_id = region_gene_id
        gene_symbol = g_info.get('gene_symbol', gene_id)
        biotype = g_info.get('biotype', '')
        annotated_strand = g_info.get('strand', '')

    tx_id = ''
    if gene_id:
        for ftype in FINE_REGION_PRIORITY:
            tx_ivs = ann.overlapping(chrom, ftype, start0, end0)
            tx_ivs = [iv for iv in tx_ivs if iv[2].get('wormbase_gene_id', '') == gene_id and iv[2].get('transcript_id', '')]
            if tx_ivs:
                tx_id = tx_ivs[0][2].get('transcript_id', '')
                break
        if not tx_id:
            for tx_ftype in ('mRNA',) + tuple(RNA_GENE_TYPES):
                tx_ivs = ann.overlapping(chrom, tx_ftype, start0, end0)
                tx_ivs = [iv for iv in tx_ivs if iv[2].get('wormbase_gene_id', '') == gene_id]
                if tx_ivs:
                    tx_id = tx_ivs[0][2].get('transcript_id', '')
                    if not biotype:
                        biotype = tx_ivs[0][2].get('biotype', tx_ftype)
                    break

    hit['wormbase_gene_id'] = gene_id
    hit['gene_symbol'] = gene_symbol
    hit['transcript_id'] = tx_id
    hit['biotype'] = biotype
    hit['gene_strand'] = annotated_strand

    if not gene_id:
        hit['broad_region'] = 'intergenic'
        hit['fine_region'] = ''
    elif final_region in ('promoter', 'TSS_region'):
        hit['broad_region'] = final_region
        hit['fine_region'] = ''
    elif final_region in FINE_REGION_PRIORITY:
        hit['broad_region'] = 'gene_body'
        hit['fine_region'] = final_region
    elif gene_overlaps:
        hit['broad_region'] = 'gene_body'
        hit['fine_region'] = ''
    else:
        hit['broad_region'] = 'intergenic'
        hit['fine_region'] = ''

    return hit

def _feature_id_from_attrs(attrs: dict, fallback: str) -> str:
    for key in ('ID', 'Name', 'locus', 'Alias'):
        if attrs.get(key):
            return _first_id(attrs[key])
    return fallback

def load_overlap_intervals(path: Path) -> dict:
    intervals = defaultdict(list)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip() or line.startswith('#') or line.startswith('track'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 3:
                cols = line.split()
            if len(cols) < 3:
                continue

            feature_id = ''
            try:
                if len(cols) >= 9 and cols[3].isdigit() and cols[4].isdigit():
                    chrom = cols[0]
                    start0, end0 = int(cols[3]) - 1, int(cols[4])
                    attrs = _gff3_attrs(cols[8])
                    feature_id = _feature_id_from_attrs(attrs, f'{cols[2]}:{chrom}:{start0 + 1}-{end0}')
                else:
                    chrom = cols[0]
                    start0, end0 = int(cols[1]), int(cols[2])
                    feature_id = cols[3] if len(cols) >= 4 else f'{chrom}:{start0 + 1}-{end0}'
            except ValueError:
                log.warning(f"Skipping malformed overlap row {path}:{line_no}")
                continue

            if end0 > start0:
                intervals[chrom].append((start0, end0, feature_id))

    for chrom in intervals:
        intervals[chrom].sort(key=lambda x: x[0])
    return intervals

def annotate_interval_overlaps(hits: list, overlap_path: Path, flag_col: str, id_col: str):
    if not overlap_path or not overlap_path.exists():
        return
    log.info(f"Loading overlaps for {flag_col}: {overlap_path}")
    intervals = load_overlap_intervals(overlap_path)

    n_flagged = 0
    for hit in hits:
        chrom = hit['chromosome']
        start0, end0 = hit['start'] - 1, hit['end']
        matched_ids = []
        for bs, be, feature_id in intervals.get(chrom, []):
            if bs >= end0:
                break
            if be > start0:
                matched_ids.append(feature_id)
        if matched_ids:
            hit[flag_col] = True
            hit[id_col] = ';'.join(dict.fromkeys(matched_ids))
            n_flagged += 1
    log.info(f"  {n_flagged:,} hits flagged for {flag_col}")

def annotate_gff3_feature_overlaps(hits: list, ann: GenomeAnnotation):
    for ftype, (flag_col, id_col) in GFF3_OVERLAP_MAP.items():
        n_flagged = 0
        for hit in hits:
            chrom = hit['chromosome']
            start0, end0 = hit['start'] - 1, hit['end']
            overlaps = ann.overlapping(chrom, ftype, start0, end0)
            if overlaps:
                ids = [iv[2].get('feature_id', f'{ftype}:{chrom}:{iv[0] + 1}-{iv[1]}') for iv in overlaps]
                hit[flag_col] = True
                hit[id_col] = ';'.join(dict.fromkeys(ids))
                n_flagged += 1
        log.info(f"  {n_flagged:,} hits flagged for {flag_col} from GFF3 {ftype}")

# ── Aggregations & Matrices ───────────────────────────────────────────────────
FINE_REGIONS = ['five_prime_UTR', 'CDS', 'intron', 'three_prime_UTR', 'exon']

def _union_length(intervals: list) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total

def precompute_region_lengths(ann: GenomeAnnotation) -> dict:
    lengths_by_gene = defaultdict(lambda: defaultdict(int))
    intervals_by_gene_region = defaultdict(lambda: defaultdict(list))

    for gene_id, g_info in ann.genes.items():
        gene_len = g_info.get('end0', 0) - g_info.get('start0', 0)
        lengths_by_gene[gene_id]['gene_body'] = gene_len
        lengths_by_gene[gene_id]['total'] = gene_len

    for chrom, ftypes in ann.features.items():
        for ftype in FINE_REGIONS + ['promoter', 'TSS_region']:
            for start0, end0, meta in ftypes.get(ftype, []):
                gene_id = meta.get('wormbase_gene_id', '')
                if gene_id:
                    intervals_by_gene_region[gene_id][ftype].append((start0, end0))

    for gene_id, by_region in intervals_by_gene_region.items():
        for ftype, intervals in by_region.items():
            lengths_by_gene[gene_id][ftype] = _union_length(intervals)

    return lengths_by_gene

def build_catalogue(all_hits: list, ann: GenomeAnnotation, gene_descriptions: dict, protein_domains: dict, genome_sequences: dict) -> list:   
    by_gene = defaultdict(lambda: {'pis': [], 'pgs': []})
    for hit in all_hits:
        gid = hit['wormbase_gene_id'] or '__intergenic__'
        mtype = hit['motif_type'].lower()
        by_gene[gid][mtype].append(hit)

    rows = []
    region_lengths_by_gene = precompute_region_lengths(ann)

    def density(count, length_bp):
        return round(count / length_bp * 1000, 4) if length_bp > 0 else 0.0

    for gene_id, hit_sets in by_gene.items():
        g_info = ann.genes.get(gene_id, {})
        gene_len = (g_info.get('end0', 0) - g_info.get('start0', 0))
        region_lengths = region_lengths_by_gene.get(gene_id, {})

        # ======================================================================
        # Nucleotide composition calculations
        # ======================================================================
        chrom = g_info.get('chrom')
        if chrom and chrom in genome_sequences:
            seq = genome_sequences[chrom][g_info['start0']:g_info['end0']].upper()
            g_count = seq.count("G")
            c_count = seq.count("C")
            a_count = seq.count("A")
            t_count = seq.count("T")
            seq_len = len(seq)
            g_percentage = round(g_count * 100 / seq_len, 2) if seq_len else 0.0
            c_percentage = round(c_count * 100 / seq_len, 2) if seq_len else 0.0
            gc_percentage = round((g_count + c_count) * 100 / seq_len, 2) if seq_len else 0.0
        else:
            g_count = c_count = a_count = t_count = 0
            g_percentage = c_percentage = gc_percentage = 0.0

        row = {
            'wormbase_gene_id': gene_id, 'gene_symbol': g_info.get('gene_symbol', gene_id),
            'gene_function': gene_descriptions.get(gene_id, ''), 'protein_domains': protein_domains.get(gene_id, ''),
            'chromosome': chrom or '', 'strand': g_info.get('strand', ''), 'biotype': g_info.get('biotype', ''),
            'gene_start': g_info.get('start0', 0) + 1, 'gene_end': g_info.get('end0', 0), 'gene_length': gene_len,
            
            'g_count': g_count,
            'c_count': c_count,
            'a_count': a_count,
            't_count': t_count,
            'g_percentage': g_percentage,
            'c_percentage': c_percentage,
            'gc_percentage': gc_percentage,
        }

        for freg in FINE_REGIONS:
            row[f'{freg}_length'] = region_lengths.get(freg, 0)
        for breg in ['promoter', 'TSS_region', 'gene_body', 'intergenic']:
            row[f'{breg}_length'] = region_lengths.get(breg, 0)

        for mtype in ('pis', 'pgs'):
            hits = hit_sets[mtype]
            total = len(hits)
            row[f'{mtype}_total'] = total
            row[f'{mtype}_total_density_per_kb'] = density(total, gene_len)

            for freg in FINE_REGIONS:
                cnt = sum(1 for h in hits if h['fine_region'] == freg)
                row[f'{mtype}_{freg}_count'] = cnt
                row[f'{mtype}_{freg}_density_per_kb'] = density(cnt, region_lengths.get(freg, 0))

            for breg in ['promoter', 'TSS_region', 'gene_body', 'intergenic']:
                cnt = sum(1 for h in hits if h['broad_region'] == breg)
                row[f'{mtype}_{breg}_count'] = cnt
                row[f'{mtype}_{breg}_density_per_kb'] = density(cnt, region_lengths.get(breg, 0))

        rows.append(row)

    rows.sort(key=lambda r: r['pgs_total'] + r['pis_total'], reverse=True)
    return rows

def build_summary(catalogue_rows: list) -> list:
    summary = []
    for row in catalogue_rows:
        gene_len = row.get('gene_length', 0)
        summary.append({
            'wormbase_gene_id': row['wormbase_gene_id'], 'gene_symbol': row['gene_symbol'],
            'gene_function': row.get('gene_function', ''), 'protein_domains': row.get('protein_domains', ''),
            'chromosome': row['chromosome'], 'strand': row['strand'], 'biotype': row['biotype'], 'gene_length': gene_len,
            'pis_total_hits': row['pis_total'], 'pis_total_density_per_kb': row['pis_total_density_per_kb'],
            'pis_CDS_hits': row.get('pis_CDS_count', 0), 'pis_five_prime_UTR_hits': row.get('pis_five_prime_UTR_count', 0),
            'pis_three_prime_UTR_hits': row.get('pis_three_prime_UTR_count', 0), 'pis_intron_hits': row.get('pis_intron_count', 0),
            'pis_promoter_hits': row.get('pis_promoter_count', 0),
            'pgs_total_hits': row['pgs_total'], 'pgs_total_density_per_kb': row['pgs_total_density_per_kb'],
            'pgs_CDS_hits': row.get('pgs_CDS_count', 0), 'pgs_five_prime_UTR_hits': row.get('pgs_five_prime_UTR_count', 0),
            'pgs_three_prime_UTR_hits': row.get('pgs_three_prime_UTR_count', 0), 'pgs_intron_hits': row.get('pgs_intron_count', 0),
            'pgs_promoter_hits': row.get('pgs_promoter_count', 0),
        })
    summary.sort(key=lambda r: r['pgs_total_hits'] + r['pis_total_hits'], reverse=True)
    return summary

def validate_outputs(all_hits: list, catalogue_rows: list):
    bad_symbols = [h for h in all_hits if h.get('gene_symbol', '').startswith('gmap:')]
    bad_intergenic = [h for h in all_hits if h.get('broad_region') == 'intergenic' and h.get('fine_region')]
    missing_gene_nonintergenic = [h for h in all_hits if not h.get('wormbase_gene_id') and h.get('broad_region') != 'intergenic']
    
    duplicate_keys = defaultdict(int)
    for h in all_hits:
        duplicate_keys[(h['chromosome'], h['start'], h['end'], h['motif_type'])] += 1
    duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)

    hit_totals = defaultdict(lambda: defaultdict(int))
    for h in all_hits:
        gid = h['wormbase_gene_id'] or '__intergenic__'
        hit_totals[gid][h['motif_type'].lower()] += 1

    catalogue_mismatches = []
    for row in catalogue_rows:
        gid = row['wormbase_gene_id']
        for mtype in ('pis', 'pgs'):
            if row.get(f'{mtype}_total', 0) != hit_totals[gid][mtype]:
                catalogue_mismatches.append((gid, mtype))

    if bad_symbols: log.warning(f"Validation: {len(bad_symbols):,} suspicious symbols remain")
    if bad_intergenic: log.warning(f"Validation: {len(bad_intergenic):,} intergenic hits have fine regions")
    if missing_gene_nonintergenic: log.warning(f"Validation: {len(missing_gene_nonintergenic):,} non-intergenic hits lack IDs")
    if duplicate_count: log.warning(f"Validation: {duplicate_count:,} duplicate intervals remain")
    if catalogue_mismatches: log.warning(f"Validation: {len(catalogue_mismatches):,} totals mismatch")
    if not any((bad_symbols, bad_intergenic, missing_gene_nonintergenic, duplicate_count, catalogue_mismatches)):
        log.info("Validation passed successfully.")

def density_stats(values):
    positive = [v for v in values if v > 0]
    return {
        "mean_all_regions": mean(values) if values else 0.0, "median_all_regions": median(values) if values else 0.0,
        "mean_positive_regions": mean(positive) if positive else 0.0, "median_positive_regions": median(positive) if positive else 0.0,
    }

def write_csv(path: Path, fieldnames: list, rows: list, label: str, tmp_dir: Path):
    log.info(f"Writing {label} → {path.name}  ({len(rows):,} rows)")
    tmp_path = tmp_dir / path.name
    with open(tmp_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    for attempt in range(1, 6):
        try:
            if path.exists(): path.unlink()
            shutil.copy2(tmp_path, path)
            tmp_path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(15 * attempt)
    raise IOError(f"Failed staging write for {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='C. elegans WBcel235 forward-strand PIS/PGS DNA scanner')
    parser.add_argument('--fasta', required=True, help='Ensembl toplevel genome FASTA (.fa or .fa.gz)')
    parser.add_argument('--gff3', required=True, help='WormBase WBPS GFF3 annotation (.gff3 or .gff3.gz)')
    parser.add_argument('--functional_descriptions', default=None, help='WormBase functional descriptions file')
    parser.add_argument('--protein_domains', default=None, help='WormBase protein domains TSV')
    parser.add_argument('--outdir', required=True, help='Output directory')
    parser.add_argument('--g_quartet_bed', default=None)
    parser.add_argument('--tf_binding_bed', default=None)
    parser.add_argument('--dnase_bed', default=None)
    parser.add_argument('--enhancer_bed', default=None)
    parser.add_argument('--limit', type=int, default=0)

    args = parser.parse_args()
    fasta_path, gff3_path, outdir = Path(args.fasta), Path(args.gff3), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ann = parse_gff3(gff3_path)
    gene_descriptions = load_functional_descriptions(args.functional_descriptions) if args.functional_descriptions else {}
    protein_domains = load_protein_domains(args.protein_domains) if args.protein_domains else {}

    all_hits = []
    n_chroms = 0
    chrom_stats = {}
    genome_sequences = {}

    log.info("Scanning genome (forward strand only) …")
    for chrom, seq in tqdm(stream_genome_fasta(fasta_path), desc="Chromosomes", unit="chr"):
        genome_sequences[chrom] = seq
        chrom_len = len(seq)
        raw_hits = scan_chromosome(chrom, seq)

        for hit in raw_hits:
            annotate_hit(hit, ann)

        all_hits.extend(raw_hits)
        n_pis = sum(1 for h in raw_hits if h['motif_type'] == 'PIS')
        n_pgs = sum(1 for h in raw_hits if h['motif_type'] == 'PGS')
        
        chrom_stats[chrom] = {
            'length_bp': chrom_len,
            'pis_hits': n_pis,
            'pgs_hits': n_pgs,
        }
        log.info(f"    Chromosome {chrom}: PIS={n_pis:,}  PGS={n_pgs:,}")

        n_chroms += 1
        if args.limit and n_chroms >= args.limit:
            break

    for hit in all_hits:
        gid = hit.get("wormbase_gene_id", "")
        hit["gene_function"] = gene_descriptions.get(gid, "")
        hit["protein_domains"] = protein_domains.get(gid, "")

    chrom_order = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "X": 5, "MtDNA": 6}
    all_hits.sort(key=lambda h: (chrom_order.get(h["chromosome"], 999), h["start"], h["end"], h["motif_type"]))

    log.info("Annotating overlap landmarks …")
    annotate_gff3_feature_overlaps(all_hits, ann)
    
    bed_map = {
        'overlaps_G_quartet': (args.g_quartet_bed, 'g_quartet_id'),
        'overlaps_TF_binding_site': (args.tf_binding_bed, 'tf_binding_site_id'),
        'overlaps_DNaseI_site': (args.dnase_bed, 'dnase_site_id'),
        'overlaps_enhancer': (args.enhancer_bed, 'enhancer_id'),
    }
    for flag_col, (overlap_arg, id_col) in bed_map.items():
        if overlap_arg:
            annotate_interval_overlaps(all_hits, Path(overlap_arg), flag_col, id_col)

    pis_hits = [h for h in all_hits if h['motif_type'] == 'PIS']
    pgs_hits = [h for h in all_hits if h['motif_type'] == 'PGS']

    catalogue_rows = build_catalogue(all_hits, ann, gene_descriptions, protein_domains, genome_sequences)
    summary_rows = build_summary(catalogue_rows)
    validate_outputs(all_hits, catalogue_rows)

    tmp_stage = Path(tempfile.gettempdir()) / f'pis_pgs_{SPECIES_TAG}'
    tmp_stage.mkdir(exist_ok=True)

    write_csv(outdir / f'PGS_Hits_{SPECIES_TAG}.csv', HITS_FIELDNAMES, pgs_hits, 'PGS Hits', tmp_stage)
    write_csv(outdir / f'PGS_Catalogue_{SPECIES_TAG}.csv', CATALOGUE_FIELDNAMES, catalogue_rows, 'PGS Catalogue', tmp_stage)
    write_csv(outdir / f'PGS_Summary_{SPECIES_TAG}.csv', SUMMARY_FIELDNAMES, summary_rows, 'PGS Summary', tmp_stage)
    write_csv(outdir / f'PIS_Hits_{SPECIES_TAG}.csv', HITS_FIELDNAMES, pis_hits, 'PIS Hits', tmp_stage)
    write_csv(outdir / f'PIS_Catalogue_{SPECIES_TAG}.csv', CATALOGUE_FIELDNAMES, catalogue_rows, 'PIS Catalogue', tmp_stage)
    write_csv(outdir / f'PIS_Summary_{SPECIES_TAG}.csv', SUMMARY_FIELDNAMES, summary_rows, 'PIS Summary', tmp_stage)

    REGIONS_MAP = {
        "promoter": ("pgs_promoter_density_per_kb", "pis_promoter_density_per_kb", "promoter_length"),
        "TSS_region": ("pgs_TSS_region_density_per_kb", "pis_TSS_region_density_per_kb", "TSS_region_length"),
        "five_prime_UTR": ("pgs_five_prime_UTR_density_per_kb", "pis_five_prime_UTR_density_per_kb", "five_prime_UTR_length"),
        "CDS": ("pgs_CDS_density_per_kb", "pis_CDS_density_per_kb", "CDS_length"),
        "intron": ("pgs_intron_density_per_kb", "pis_intron_density_per_kb", "intron_length"),
        "three_prime_UTR": ("pgs_three_prime_UTR_density_per_kb", "pis_three_prime_UTR_density_per_kb", "three_prime_UTR_length"),
        "exon": ("pgs_exon_density_per_kb", "pis_exon_density_per_kb", "exon_length"),
        "gene_body": ("pgs_gene_body_density_per_kb", "pis_gene_body_density_per_kb", "gene_body_length"),
        "total_gene": ("pgs_total_density_per_kb", "pis_total_density_per_kb", "gene_length"),
    }
    
    region_density_statistics = {}
    for region, (pgs_col, pis_col, len_col) in REGIONS_MAP.items():
        pgs = [r.get(pgs_col, 0) for r in catalogue_rows if r.get(len_col, 0) > 0]
        pis = [r.get(pis_col, 0) for r in catalogue_rows if r.get(len_col, 0) > 0]
        region_density_statistics[region] = {"pgs": density_stats(pgs), "pis": density_stats(pis)}

    stats = {
        'species': 'Caenorhabditis elegans', 'assembly': 'WBcel235', 'annotation_source': 'WormBase WBPS19 (PRJNA13758)',
        'scan_mode': 'genome-wide forward-strand DNA scan', 'chromosomes_scanned': n_chroms,
        'total_pis_hits': len(pis_hits), 'total_pgs_hits': len(pgs_hits), 'genes_annotated': len(ann.genes),
        'pis_pattern': PIS_PATTERN.pattern, 'pgs_pattern': PGS_PATTERN.pattern,
        'per_chromosome': chrom_stats, 'region_density_statistics': region_density_statistics,
    }
    
    stats_path = outdir / f'scan_stats_{SPECIES_TAG}.json'
    with open(tmp_stage / stats_path.name, 'w') as fh:
        json.dump(stats, fh, indent=2)
    shutil.copy2(tmp_stage / stats_path.name, stats_path)
    (tmp_stage / stats_path.name).unlink(missing_ok=True)
    
    log.info("=" * 60 + "\nALL DONE\n" + "=" * 60)

if __name__ == '__main__':
    main()