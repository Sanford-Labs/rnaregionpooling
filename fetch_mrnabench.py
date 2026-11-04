"""fetch_mrnabench.py KEY... — mRNABench datasets from Hugging Face, written as the tables the scripts here read.

mRNABench (`morrislab/mRNABench`, AGPL-3.0) publishes each dataset as a parquet file and cuts it into train,
validation and test sets with its own splitters. This script downloads the parquet files, rewrites each row as
5'UTR, CDS and 3'UTR sequences with their splice-site coordinates, and adds the split of mRNABench's default seed,
without importing `mrna_bench`: `split_homologous` and `split_random` are ports of `mrna_bench.data_splitter`'s
homology and random splitters, and the homology groups are the HomoloGene maps the mRNABench authors published
on Zenodo.

The `cds` column of mRNABench is a codon-frame track: 1 on the first base of every codon and 0 elsewhere. The CDS
runs from the first 1 to three bases past the last one. Its `splice` column marks the last base of each exon and is
written as three `{REGION}_splice_coord` columns of positions counted within each region. A row without an
annotated CDS keeps its nucleotides (`--noncoding` says in which region) and `has_cds` marks it.

Usage:
    python fetch_mrnabench.py --list
    python fetch_mrnabench.py rnahl-human translation-efficiency-human ...
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA

HF = "https://huggingface.co/datasets/morrislab"

# The homology groups every `homology` split is cut along: HomoloGene, processed by the mRNABench
# authors and published beside the paper. It is data rather than code, which is why the split can
# be reproduced without their package at all.
HOMOLOGY_URL = "https://zenodo.org/records/13910050/files/homology_maps_homologene.zip"

# Their DataSplitter's defaults, and the seed their own scripts pass. Changing either produces a
# split that is ours and not theirs, which is fine as long as nobody then compares the numbers.
SEED = 2541
RATIOS = (0.7, 0.15, 0.15)


@dataclass(frozen=True)
class Dataset:
    """One catalogue entry: where the rows are and how they are cut.

    Attributes:
        url: The Hugging Face parquet mRNABench itself downloads. Their dataset class holds this
            string and nothing else that matters here — the rest of it processes the raw source
            with GenomeKit, which is the path this script exists to avoid.
        species: Which homology map the split reads, and empty for the tables that take the random
            one — a designed 5'UTR library has no genes and no homologues, which is why mRNABench
            offers those tables no homology split either.
        targets: The label columns, as they arrive and under the names they arrive under. Copied
            through rather than renamed: the name is what their leaderboard reports per target —
            `mrl-sample-egfp` is scored three times, once per nucleotide modification — so a column
            called something else is one nobody can line up against a published number. The tables
            that carry a single label already call it `target`, which is why the five written
            before this field was a tuple are unchanged by it.
        desc: A sentence for `--list`.
    """

    url: str
    species: str
    targets: tuple[str, ...]
    desc: str

    @property
    def homologous(self) -> bool:
        """Whether this table's split is cut along homology groups rather than at random."""
        return bool(self.species)


# A label is one number per row. A table whose label is a vector in one cell (GO terms, compartments) is
# written out one column per entry (`target_000`, ...); a table with several labels keeps one column each.
CATALOG: dict[str, Dataset] = {
    "rnahl-human": Dataset(
        f"{HF}/rnahl-saluki-human/resolve/main/rnahl-saluki-human.parquet",
        "human", ("target",), "RNA half-life, human (Agarwal & Kelley's Saluki table)"),
    "rnahl-mouse": Dataset(
        f"{HF}/rnahl-saluki-mouse/resolve/main/rnahl-saluki-mouse.parquet",
        "mouse", ("target",), "RNA half-life, mouse"),
    "translation-efficiency-human": Dataset(
        f"{HF}/translation-efficiency-human/resolve/main/te_human.parquet",
        "human", ("target",),
        "Translation efficiency, human (Nat Biotech 2025 s41587-025-02712-x)"),
    "translation-efficiency-mouse": Dataset(
        f"{HF}/translation-efficiency-mouse/resolve/main/te_mouse.parquet",
        "mouse", ("target",), "Translation efficiency, mouse"),
    "mrl-sugimoto": Dataset(
        f"{HF}/mrl-sugimoto/resolve/main/mrl-sugimoto.parquet",
        "human", ("target",), "Mean ribosome load per isoform (Sugimoto)"),
    # ── The designed libraries, whose split is random because nothing else is available ──────────
    #
    # Every row here is a fixed reporter construct with a variable 5'UTR — a molecule somebody
    # ordered, not a transcript somebody found. So there is no gene to look up in a homology map,
    # and mRNABench does not offer these tables a homology split at all: its leaderboard reports
    # them under `default`, which is its random split, and under two k-mer splits. Only `default`
    # is ported. The k-mer ones hold out by sequence similarity, they are a second port with its own
    # ways of being subtly not-theirs, and `default` is the one every model on their board has.
    "mrl-sample-egfp": Dataset(
        f"{HF}/mrl-sample/resolve/main/mrl-sample-egfp.parquet", "",
        ("target_mrl_egfp_unmod", "target_mrl_egfp_pseudo", "target_mrl_egfp_m1pseudo"),
        "Mean ribosome load, Sample MPRA random 5'UTR library, eGFP (3 modifications)"),
    "mrl-sample-mcherry": Dataset(
        f"{HF}/mrl-sample/resolve/main/mrl-sample-mcherry.parquet", "",
        ("target_mrl_mcherry",), "Mean ribosome load, Sample MPRA random library, mCherry"),
    "mrl-sample-designed": Dataset(
        f"{HF}/mrl-sample/resolve/main/mrl-sample-designed.parquet", "",
        ("target_mrl_designed",), "Mean ribosome load, Sample MPRA designed 5'UTRs"),
    "mrl-sample-varying": Dataset(
        f"{HF}/mrl-sample/resolve/main/mrl-sample-varying.parquet", "",
        ("target_mrl_varying_length",),
        "Mean ribosome load, Sample MPRA library of varying 5'UTR length"),
    "mrl-hl-lbkwk": Dataset(
        f"{HF}/mrl-hl-lbkwk/resolve/main/mrl-hl-lbkwk.parquet", "",
        ("target_ribosome_load", "target_in_cell_half_life"),
        "Ribosome load AND half-life over the same 203 designed mRNAs (Leppek 2022)"),
    # ── The classification tables of their Table 2 ───────────────────────────────────────────────
    #
    # eCLIP carries one 0/1 column per RBP and the board probes the twenty with the most events
    # per cell line (`ECLIP_*_TOP_RBPS_LIST` in their `eclip_binding.py`); those twenty are the
    # targets here. The other five carry ONE `target` column holding a 0/1 vector — GO terms,
    # compartments — and `convert` writes that vector out as `target_000`, `target_001`, ... so
    # each label is a column of its own. VEP is the linear-probe form of TraitGym: two rows per
    # variant, the reference sequence labelled 0 and the variant its pathogenicity, split at random
    # as their catalogue has it.
    "eclip-binding-k562": Dataset(
        f"{HF}/eclip/resolve/main/eclip-k562.parquet", "human",
        tuple(f"target_{rbp}" for rbp in (
            "YBX3", "UCHL5", "ZNF622", "DDX3X", "LIN28B", "PUM2", "PABPC4", "DDX24",
            "IGF2BP1", "IGF2BP2", "RBM15", "FAM120A", "PUM1", "SND1", "DDX6", "METAP2",
            "FXR2", "PCBP1", "TIA1", "FMR1")),
        "eCLIP binding, K562: the top 20 RBPs by events, one 0/1 column each"),
    "eclip-binding-hepg2": Dataset(
        f"{HF}/eclip/resolve/main/eclip-hepg2.parquet", "human",
        tuple(f"target_{rbp}" for rbp in (
            "PPIG", "DDX3X", "LARP4", "LIN28B", "G3BP1", "NCBP2", "IGF2BP1", "AKAP1",
            "PCBP2", "PABPN1", "SND1", "UCHL5", "DDX55", "FXR2", "EIF3H", "IGF2BP3",
            "SRSF1", "HLTF", "LSM11", "PRPF4")),
        "eCLIP binding, HepG2: the top 20 RBPs by events, one 0/1 column each"),
    "go-bp": Dataset(
        f"{HF}/go-bp/resolve/main/go_dna_dataset_bp.parquet", "human", ("target",),
        "GO biological process, 20 terms, multilabel"),
    "go-cc": Dataset(
        f"{HF}/go-cc/resolve/main/go_dna_dataset_cc.parquet", "human", ("target",),
        "GO cellular component, 20 terms, multilabel"),
    "go-mf": Dataset(
        f"{HF}/go-mf/resolve/main/go_dna_dataset_mf.parquet", "human", ("target",),
        "GO molecular function, 20 terms, multilabel"),
    "prot-loc": Dataset(
        f"{HF}/protein-localization/resolve/main/prot-loc.parquet", "human", ("target",),
        "Protein localisation, 12 compartments, multilabel"),
    "rna-loc-fazal": Dataset(
        f"{HF}/rna-loc-fazal/resolve/main/rna-loc-fazal.parquet", "human", ("target",),
        "mRNA localisation, APEX-seq (Fazal 2019), 8 compartments, multilabel"),
    "rna-lifecycle-ietswaart": Dataset(
        f"{HF}/rna-lifecycle-ietswaart/resolve/main/ietswaart_processed.parquet", "human",
        ("target",), "mRNA localisation, subcellular RNA lifecycle (Ietswaart 2024), 3 labels"),
    "vep-traitgym-complex": Dataset(
        f"{HF}/vep-traitgym-mrna/resolve/main/vep-traitgym-complex.parquet", "",
        ("target",), "Variant effect, TraitGym complex traits, UTR variants, 0/1"),
    "vep-traitgym-mendelian": Dataset(
        f"{HF}/vep-traitgym-mrna/resolve/main/vep-traitgym-mendelian.parquet", "",
        ("target",), "Variant effect, TraitGym Mendelian traits, UTR variants, 0/1"),
}

STOPS = frozenset({"TAA", "TAG", "TGA"})


def download(url: str, dest: Path) -> Path:
    """Fetch `url` to `dest`, once.

    Args:
        url: What to get.
        dest: Where it goes. An existing file is left alone — these are immutable published
            artifacts, so re-fetching one can only cost time.

    Returns:
        `dest`.

    Raises:
        urllib.error.URLError: The fetch failed. Nothing partial is left behind: the download lands
            on `.part` and is renamed only once it is whole.
    """
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {url}", file=sys.stderr)
    request = urllib.request.Request(url, headers={"User-Agent": "rnaregionpooling/fetch_mrnabench"})
    with urllib.request.urlopen(request) as response, part.open("wb") as out:
        shutil.copyfileobj(response, out)
    part.rename(dest)
    return dest


def homology_map(species: str, cache: Path) -> pd.DataFrame:
    """The gene → homology-group table the `homology` split is cut along.

    Args:
        species: `human`, `mouse`, … — the zip carries ten and each is one CSV.
        cache: Where the zip is unpacked to.

    Returns:
        A frame with `gene_name` and `gene_group`, as mRNABench's splitter expects it.

    Raises:
        FileNotFoundError: The zip holds no map for this species.
    """
    directory = cache / "homology_maps"
    csv = directory / f"{species}_homology_map.csv"
    if not csv.exists():
        archive = download(HOMOLOGY_URL, cache / "homology_maps_homologene.zip")
        with zipfile.ZipFile(archive) as zipped:
            for name in zipped.namelist():
                if name.endswith("_homology_map.csv"):
                    directory.mkdir(parents=True, exist_ok=True)
                    with zipped.open(name) as source, (directory / Path(name).name).open("wb") as f:
                        shutil.copyfileobj(source, f)
    if not csv.exists():
        raise FileNotFoundError(f"no homology map for {species!r} in {HOMOLOGY_URL}")
    return pd.read_csv(csv)


def split_homologous(genes: list[str], homology: pd.DataFrame,
                     test_size: float, seed: int) -> tuple[list[int], list[int]]:
    """Two sets of row indices, with homologous genes kept on one side.

    A verbatim port of `mrna_bench.data_splitter.homology_split.train_test_split_homologous`,
    including the things about it that are surprising, because a split that is nearly theirs is
    not theirs: it walks a shuffled index, and the first row of a homology group drags every other
    row of that group along with it, so groups never straddle. It seeds the LEGACY global numpy RNG
    rather than a `Generator`, which is what makes the shuffle reproducible against their runs.

    Its one soft spot is worth knowing before reading a number off it. A gene the map does not
    cover has no group, so it travels alone — and on an isoform-level table like `mrl-sugimoto`,
    where 11693 rows carry only 7392 genes, two isoforms of one uncovered gene can land on
    opposite sides. The map covers about half the genes of these tables, so "homology split" is a
    weaker statement than it sounds.

    Args:
        genes: One gene name per row, in row order.
        homology: `gene_name` → `gene_group`, from `homology_map`.
        test_size: The fraction going to the second set.
        seed: Seeds `numpy.random`, globally, as theirs does.

    Returns:
        `(train_indices, test_indices)` — positions into `genes`, together covering every row.
    """
    groups_of = homology.set_index("gene_name")["gene_group"].to_dict()
    group = np.array([groups_of.get(gene, None) for gene in genes])

    rows_of_group: dict[object, list[int]] = {}
    for i, g in enumerate(group):
        if g is not None:
            rows_of_group.setdefault(g, []).append(i)

    order = np.arange(len(genes))
    np.random.seed(seed)
    np.random.shuffle(order)

    n_train = int(len(order) * (1 - test_size))
    train: list[int] = []
    test: list[int] = []
    seen: set[object] = set()
    for i in order:
        g = group[i]
        if g is not None and g in seen:
            continue
        seen.add(g)
        (train if len(train) < n_train else test).extend(rows_of_group.get(g, [int(i)]))
    return [int(i) for i in train], [int(i) for i in test]


def split_random(rows: int, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    """Two sets of row indices, cut at random — `mrna_bench`'s `default` split.

    A port of `sklearn.model_selection.train_test_split(df, test_size=…, random_state=…)`, which
    is the whole of their `SklearnSplitter.split_df`, down to the three details that decide which
    row lands where:

        the RNG           `check_random_state(seed)` is a legacy `RandomState`, not a `Generator`,
                          and the two draw different permutations from the same integer
        the SIZE          `n_test = ceil(test_size * n)` — ceil, so 15% of 203 is 31 and not 30
        the ORDER         the test half is the permutation's HEAD and the train half its tail, and
                          each comes back in permuted order rather than sorted. That is not
                          cosmetic here: the second cut of a three-way split runs on the frame the
                          first one returned, so sorting between them would hand it a different
                          table and produce a val/test division that is nobody's.

    Ported rather than imported for the reason at the top of this file — nothing here may pull
    `mrna_bench`'s pinned stack in — and checked rather than trusted: `_check_random_split` runs
    both against each other when scikit-learn happens to be importable.

    Args:
        rows: How many rows there are.
        test_size: The fraction going to the second set.
        seed: `random_state`.

    Returns:
        `(train_indices, test_indices)` — positions into the table, in the order sklearn would
            return them, together covering every row.
    """
    n_test = int(np.ceil(test_size * rows))
    order = np.random.RandomState(seed).permutation(rows)
    return [int(i) for i in order[n_test:]], [int(i) for i in order[:n_test]]


def three_way(rows: int, cut, ratios: tuple[float, float, float], seed: int) -> np.ndarray:
    """A `train` / `val` / `test` label per row.

    Their `DataSplitter.get_all_splits_df` in the shape this script needs: one split off the whole
    table, then a second split of what is left over, and the answer written back as a column rather
    than as three frames.

    Both splitters compose through here identically because their base class composes them
    identically: `get_all_splits_df` is written once and calls whichever `split_df` it was built
    with, so a second copy of this arithmetic for the random tables would be a second chance to get
    the leftover-index carry wrong.

    Args:
        rows: How many rows there are.
        cut: `(positions, test_size, seed) -> (first, second)`, all three in positions into the
            TABLE — `cutter`'s answer, over whichever split this table has.
        ratios: Train, validation, test fractions; must sum to 1.
        seed: Passed to both splits, as theirs is.

    Returns:
        An array of `train` / `val` / `test`, one per row, positionally aligned to the table.

    Raises:
        ValueError: `ratios` does not sum to 1, or the two cuts between them left a row with no
            label — which a selector would then drop in silence, training on fewer rows than the
            table holds and reporting nothing about it.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios} summing to {sum(ratios)}")

    held_out = ratios[1] + ratios[2]
    train, rest = cut(list(range(rows)), held_out, seed)
    # The second cut runs on the leftover ROWS, so its indices are positions in `rest` and have to
    # be carried back through it. Getting this wrong labels the right number of rows in the wrong
    # places, which no count ever shows. `cutter` is where that carry lives, so that both splitters
    # get it from one place.
    val, test = cut(rest, ratios[2] / held_out, seed)

    label = np.empty(rows, dtype=object)
    label[train] = "train"
    label[val] = "val"
    label[test] = "test"
    if (unlabelled := int(pd.isna(label).sum())) != 0:
        raise ValueError(f"{unlabelled} row(s) came out of the split with no label")
    return label


def cutter(genes: list[str] | None, homology: pd.DataFrame | None):
    """The `cut` `three_way` takes, over whichever split this table has.

    Args:
        genes: One gene name per row, in row order, or `None` for a table with no genes — a designed
            library, which takes the random split.
        homology: `gene_name` → `gene_group`, or `None` with `genes`.

    Returns:
        A callable taking `(positions, test_size, seed)` and returning two lists of POSITIONS INTO
            THE TABLE. Both splitters natively answer in positions into what they were handed, so
            the mapping back through `positions` happens here, once, for both.
    """
    def cut(positions: list[int], test_size: float,
            seed: int) -> tuple[list[int], list[int]]:
        if genes is None or homology is None:
            first, second = split_random(len(positions), test_size, seed)
        else:
            first, second = split_homologous([genes[i] for i in positions], homology,
                                             test_size, seed)
        return [positions[i] for i in first], [positions[i] for i in second]

    return cut


def regions(sequence: str, codon_starts: np.ndarray,
            extend_stop: bool = False) -> tuple[str, str, str] | None:
    """One mature mRNA cut into 5'UTR, CDS and 3'UTR.

    Args:
        sequence: The nucleotides, as mRNABench wrote them (DNA alphabet).
        codon_starts: Their `cds` column — 1 on each codon's first base, 0 elsewhere.
        extend_stop: Take the following codon into the CDS when it is a stop, for the tables whose
            annotation ends before theirs. `stop_convention` decides this per table; doing it per
            ROW would move a boundary every time a 3'UTR happened to open on a stop codon, which is
            one nucleotide triplet in twenty-one by chance alone.

    Returns:
        `(utr5, cds, utr3)`, concatenating back to `sequence` exactly, or `None` when the track is
            all zeros and the transcript therefore has no annotated CDS.
    """
    ones = np.flatnonzero(np.asarray(codon_starts))
    if ones.size == 0:
        return None
    start, end = int(ones[0]), int(ones[-1]) + 3   # the last 1 opens the last codon
    if extend_stop and sequence[end:end + 3] in STOPS:
        end += 3
    return sequence[:start], sequence[start:end], sequence[end:]


def marked(track: object, lengths: tuple[int, ...]) -> list[str]:
    """One per-position track, cut into regions and written as marked positions.

    Args:
        track: A per-position array over the whole mature sequence — mRNABench's `splice`, which is
            int8 and one entry per nucleotide.
        lengths: The regions' lengths in order, summing to the sequence's. The cut is taken from
            the LENGTHS of the strings `regions` produced rather than from the boundaries it
            computed, so the track is cut exactly where the sequence was — including the row whose
            CDS was never annotated and whose nucleotides all went into one region.

    Returns:
        One comma-separated list of positions per region, each counted from that region's own
            first nucleotide, ready to be written as a `{REGION}_splice_coord` column. Empty where
            a region carries no mark, which reads back as a track of zeros — the true statement
            that nothing is marked there.
    """
    dense = np.asarray(track)
    out: list[str] = []
    at = 0
    for n in lengths:
        out.append(",".join(str(int(i)) for i in np.flatnonzero(dense[at:at + n])))
        at += n
    return out


def stop_convention(raw: pd.DataFrame) -> bool:
    """Whether this table's CDS annotation stops one codon short of the stop codon.

    mRNABench is not of one mind about this and cannot be, because its tables come from different
    pipelines: the ones it processes itself with GenomeKit take the stop codon into the CDS
    (`translation-efficiency-human`, 99.94% of rows end on one), while the Saluki half-life tables
    it republishes end just before it (`rnahl-human`, 99.62% have the stop as the NEXT codon). The
    convention is measured per table and normalised onto the one that includes the stop.

    Args:
        raw: The frame as Hugging Face served it.

    Returns:
        `True` when the annotation excludes the stop codon and `regions` should take it back in.
    """
    included = excluded = 0
    for row in raw.itertuples(index=False):
        ones = np.flatnonzero(np.asarray(row.cds))
        if ones.size == 0:
            continue
        end = int(ones[-1]) + 3
        included += row.sequence[end - 3:end] in STOPS
        excluded += row.sequence[end:end + 3] in STOPS
    return excluded > included


def convert(key: str, raw: pd.DataFrame, homology: pd.DataFrame | None,
            ratios: tuple[float, float, float], seed: int, noncoding: str) -> pd.DataFrame:
    """One raw mRNABench frame in the layout of this repository's tables.

    Args:
        key: The catalogue key, for messages.
        raw: What Hugging Face served.
        homology: The species' gene → group map, or `None` for a designed library — no genes, no
            map, and the random split.
        ratios: Train, validation, test fractions.
        seed: The split seed.
        noncoding: Where a transcript with no annotated CDS puts its nucleotides — `utr5`, `cds`,
            or `drop` to leave it out, which makes the test set no longer theirs.

    Returns:
        A frame of the three `*_sequence` columns, the three `*_splice_coord` columns, `has_cds`,
            one column per declared target under its own name, and `split` — plus `gene`,
            `chromosome` and `transcript_id` for the sources that carry them. The three are absent
            rather than filled: a designed 5'UTR on a fixed reporter has no gene, and a column of
            `""` for it is a value a selector can match on and a person can misread as one.

    Raises:
        ValueError: A reconstructed row does not concatenate back to its source sequence, or
            `noncoding` is not one of the three.
    """
    if noncoding not in ("utr5", "cds", "drop"):
        raise ValueError(f"--noncoding is utr5, cds or drop; got {noncoding!r}")

    targets = CATALOG[key].targets
    carried = [name for name in ("gene", "chromosome", "transcript_id", "description")
               if name in raw.columns]
    extend_stop = stop_convention(raw)
    if extend_stop:
        print("  this table's CDS annotation excludes the stop codon; taking it back in",
              file=sys.stderr)
    rows = []
    n_noncoding = 0
    for at, row in enumerate(raw.itertuples(index=False)):
        cut = regions(row.sequence, row.cds, extend_stop)
        has_cds = cut is not None
        if not has_cds:
            n_noncoding += 1
            if noncoding == "drop":
                continue
            # Not a guess dressed up as an annotation: `has_cds` says which rows these are, so a
            # config that wants them gone can select them out and a reader is never misled.
            cut = (row.sequence, "", "") if noncoding == "utr5" else ("", row.sequence, "")
        utr5, cds, utr3 = cut
        if utr5 + cds + utr3 != row.sequence:
            # By POSITION, because a designed library has no name to blame it on.
            raise ValueError(f"{key}: row {at} does not reconstruct")
        splice = marked(row.splice, (len(utr5), len(cds), len(utr3)))
        record = {name: getattr(row, name) for name in carried}
        record |= {
            "UTR5_sequence": utr5,
            "CDS_sequence": cds,
            "UTR3_sequence": utr3,
            "UTR5_splice_coord": splice[0],
            "CDS_splice_coord": splice[1],
            "UTR3_splice_coord": splice[2],
            "has_cds": has_cds,
        }
        for name in targets:
            value = getattr(row, name)
            if np.ndim(value) == 0:
                record[name] = float(value)
            else:
                # A vector label, one column per entry, zero-padded so the columns sort in order.
                record |= {f"{name}_{i:03d}": float(v) for i, v in enumerate(value)}
        record["row_id"] = at
        rows.append(record)

    out = pd.DataFrame(rows)
    genes = out["gene"].tolist() if "gene" in out.columns else None
    out["split"] = three_way(len(out), cutter(genes, homology), ratios, seed)

    if n_noncoding:
        what = "dropped" if noncoding == "drop" else f"put in {noncoding}"
        print(f"  {n_noncoding} row(s) have no annotated CDS: {what}", file=sys.stderr)
    return out


def report(key: str, out: pd.DataFrame) -> None:
    """What came out, in the terms a wrong conversion would show up in.

    Args:
        key: The catalogue key.
        out: The converted frame.
    """
    counts = out["split"].value_counts()
    lengths = {name: out[f"{name}_sequence"].str.len() for name in ("UTR5", "CDS", "UTR3")}
    coding = out[out["has_cds"]]
    starts = coding["CDS_sequence"].str[:3].eq("ATG").mean()
    ends = coding["CDS_sequence"].str[-3:].isin(STOPS).mean()
    frames = coding["CDS_sequence"].str.len().mod(3).eq(0).mean()

    # Counted rather than trusted, because a splice track that came out empty is a column of zeros
    # — a legal value on it, so nothing downstream can tell that the channel is not there.
    marks = sum(out[f"{name}_splice_coord"].map(lambda c: len(c.split(",")) if c else 0)
                for name in ("UTR5", "CDS", "UTR3"))

    genes = f", {out['gene'].nunique():,} genes" if "gene" in out.columns else ""
    print(f"  {len(out):,} rows{genes}   "
          f"train {counts.get('train', 0):,} / val {counts.get('val', 0):,} / "
          f"test {counts.get('test', 0):,}")
    for name, length in lengths.items():
        print(f"    {name:4s} p10={int(length.quantile(.1)):>6,} p50={int(length.median()):>6,} "
              f"p90={int(length.quantile(.9)):>6,} max={length.max():>6,}")
    print(f"    splice: {marks.mean():.1f} marked position(s) per row, "
          f"{int((marks == 0).sum()):,} row(s) with none")
    print(f"    of the {len(coding):,} coding rows: {starts:.2%} open on ATG, "
          f"{ends:.2%} close on a stop, {frames:.2%} are in frame")
    written = [c for c in out.columns if c == "target" or c.startswith("target_")]
    if len(written) > 3:
        rates = out[written].mean()
        print(f"    {len(written)} label columns {written[0]} .. {written[-1]}: "
              f"positive rate min={rates.min():.3f} median={rates.median():.3f} "
              f"max={rates.max():.3f}")
    else:
        for name in written:
            column = out[name]
            print(f"    {name}: mean={column.mean():.4g} std={column.std():.4g} "
                  f"min={column.min():.4g} max={column.max():.4g}")


def main(argv: list[str] | None = None) -> int:
    """Fetch, convert and write each named dataset.

    Args:
        argv: Command line, or `None` for `sys.argv`.

    Returns:
        A process exit status: 0, or 1 if any dataset failed.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("keys", nargs="*", metavar="KEY", help="datasets to fetch")
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument("--out-dir", type=Path, default=DATA,
                        help="where the tables go (default: %(default)s)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="where downloads are kept (default: OUT_DIR/.cache)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="split seed; mRNABench's own is %(default)s")
    parser.add_argument("--ratios", type=float, nargs=3, default=RATIOS,
                        metavar=("TRAIN", "VAL", "TEST"), help="split fractions")
    parser.add_argument("--noncoding", choices=("utr5", "cds", "drop"), default="utr5",
                        help="where a row with no annotated CDS goes (default: %(default)s)")
    args = parser.parse_args(argv)

    if args.list or not args.keys:
        width = max(len(k) for k in CATALOG)
        for key, dataset in CATALOG.items():
            split = dataset.species or "random"
            print(f"{key:{width}s}  {split:6s}  {dataset.desc}")
        return 0 if args.list else 1

    unknown = [k for k in args.keys if k not in CATALOG]
    if unknown:
        print(f"not in the catalogue: {', '.join(unknown)}\n"
              f"available: {', '.join(CATALOG)}", file=sys.stderr)
        return 1

    cache = args.cache_dir or args.out_dir / ".cache"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for key in args.keys:
        print(f"{key}:")
        dataset = CATALOG[key]
        try:
            raw = pd.read_parquet(download(dataset.url, cache / f"{key}.raw.parquet"))
            # Only for the tables cut along it. The map is a 40 MB zip off Zenodo and a designed
            # library has nothing to look up in it.
            homology = homology_map(dataset.species, cache) if dataset.homologous else None
            out = convert(key, raw, homology, tuple(args.ratios), args.seed, args.noncoding)
        except (OSError, ValueError, KeyError) as e:
            print(f"  FAILED  {type(e).__name__}: {e}", file=sys.stderr)
            failed += 1
            continue
        destination = args.out_dir / f"{key}.parquet"
        out.to_parquet(destination, index=False)
        report(key, out)
        print(f"    -> {destination}  ({destination.stat().st_size / 1e6:.1f} MB)")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
