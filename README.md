# Region-aware pooling for mRNA foundation models

The code behind *Regional Signals Matter: Rethinking Global Pooling for mRNA Foundation Models*.

A frozen mRNA model is run once over a transcript. Instead of averaging its token embeddings over the whole
sequence, we average them within the 5′ UTR, the CDS and the 3′ UTR and concatenate the three means, then fit
the same linear probe the benchmark fits. Nothing is trained, and the readout works with any model whose
output positions can be mapped back to nucleotides.

## Environments

The fits, summaries, tables and figures run in the environment of this directory; the embeddings run in one
environment per model family, under `envs/`. All are [uv](https://docs.astral.sh/uv/) projects with a lock file.

    uv sync                                  # fits, analyses, tables, figures (Python 3.14, torch 2.13, CUDA 13)
    uv run playwright install chromium       # the browser the figure export prints with
    uv sync --project envs/multimolecule     # embeddings: multimolecule models, Helix-mRNA, DNABERT-2/-S, NT, HyenaDNA
    uv sync --project envs/evo2              # Evo 2 (builds flash-attn)
    uv sync --project envs/evo1              # Evo 1 (builds flash-attn)

## Directories

Every script reads and writes the directories named in `paths.py`:

| name | what | environment variable | default |
|---|---|---|---|
| `CODE` | this repository | — | — |
| `WORK` | embeddings, fits, element tables, perturbation scores (959 GB with the embeddings) | `REGIONS_WORK` | `work/` |
| `DATA` | the mRNABench tables | `MRNA_BENCH_DATA` | `WORK/data` |
| `MODEL_DIR` | weights not fetched from Hugging Face at run time | `REGIONS_MODELS` | `WORK/models` |
| `PAPER` | the figures and LaTeX tables | `REGIONS_PAPER` | `WORK/paper` |

## The pipeline

**0. Data.** `fetch_mrnabench.py` downloads mRNABench's datasets from Hugging Face and writes each as a table of
5′ UTR, CDS and 3′ UTR sequences, splice-site coordinates, targets and mRNABench's default split, without
importing `mrna_bench`. It takes mRNABench's dataset names (`--list` prints them); the other scripts take the
short names of `cpu_probe.KEYS`.

    uv run python fetch_mrnabench.py rnahl-human rnahl-mouse translation-efficiency-human ...

**1. Embed.** One script per model family, each writing `WORK/results/external/<model>/<table>/features/`
with one `.npy` per readout (`mbmean`, `bases`, `regions`, `split3`, …) and a `meta.json`.

| script | models | environment |
|---|---|---|
| `harness/embed_mm.py` | RNA-FM, RNA-MSM, RNAErnie, ERNIE-RNA, RiNALMo, SpliceBERT, UTR-LM, AIDO.RNA | `envs/multimolecule` |
| `harness/embed_mm2.py` | RNABERT, 3UTRBERT, DNABERT-2, DNABERT-S; RiNALMo in windows (`--window`, `--stride`) or as a token cache (`--tokens`) | `envs/multimolecule` |
| `harness/embed_hf.py` | Nucleotide Transformer, HyenaDNA | `envs/multimolecule` |
| `harness/embed_helix.py` | Helix-mRNA (`--no-e` drops the codon marks) | `envs/multimolecule` |
| `harness/embed_evo1.py`, `harness/embed_evo2.py` | Evo 1, Evo 2 | `envs/evo1`, `envs/evo2` |
| `harness/aido_model.py` | AIDO.RNA's loader | — |

`--isolate` encodes each region on its own; `--layer N` (embed_mm.py) takes layer N instead of the last;
`--parquet` embeds the edited sequences of a perturbation table instead of the benchmark's. For example:

    uv run --project envs/multimolecule python harness/embed_mm.py rinalmo-giga rnahl-human \
        work/results/external/rinalmo-giga/rnahl-human --device cuda:0 --dtype bf16

**2. Fit.** `cpu_probe.py` holds the splits and the ridge probe (float64). `harness/fit_ext.py` writes `fit.json`
per table, one entry per readout and penalty grid, ten splits each; `harness/fit_cls.py` writes `fit_cls.json` for
the classification tables, scored by `metrics.py` (AUPRC); `harness/fit_controls.py` adds the control readouts
(equal thirds, balanced mean, PCA, random and shifted boundaries, length and GC); `harness/decomp_ext.py` the
per-region variance shares; `harness/attn_pool.py` the learned attention pooling and the share of its weight on
each region.

    uv run python harness/fit_ext.py work/results/external/rinalmo-giga/rnahl-human/features

Grid `rawA` is mRNABench's penalty range (1e-3 … 10) and `rawB` ours (1e-6 … 1e6). Every number the paper
reports for the benchmark's own readout uses `mbmean|rawA`, as published.

**3. Analyse.**

| script | writes |
|---|---|
| `elements.py`, `element_compare.py` | element counts per transcript, and the measured against predicted group differences |
| `uaug_subsets.py`, `uaug_truth.py`, `uaug_truth_tables.py` | the uAUG classes and their measured effects |
| `perturb_design.py` | the edited sequences (uAUG insertion and removal, codon recoding, ARE and let-7 insertions, matched controls) |
| `perturb_score.py` | each probe's response to every edit, in units of the label's SD |
| `perturb_summary.py` | the numbers the text quotes from those scores |
| `calib_perseed.py` | the reproduction against mRNABench's published per-split scores |
| `naive_ridge.py`, `naive_cls.py`, `naive_mpra.py` | mRNABench's naive baselines |

**4. Report.** `final_tables.py` prints every summary table; `story_html.py` builds the tables and figures into
one page, `extra_figs.py` and `draft_render.py` the paper's figures and LaTeX tables, through
`draft_figs_export.py` (headless Chromium).

    uv run python final_tables.py > work/final-tables.txt

**Running it at scale.** `harness/queue_all.py --card N` works through the model × table list, one lock
directory per task, and `harness/fit_queue.py --card N` fits whatever has features and no fit yet.
`harness/watch_failed.py` stops on the first failure, `harness/reap_locks.py` frees the locks of jobs that died,
`harness/qstatus.py` prints what is left.

## Inputs fetched separately

- Model weights come from each model's Hugging Face release at run time, except AIDO.RNA's safetensors
  (genbio-ai/AIDO.RNA-1.6B and -650M, under `MODEL_DIR/aido/`) and the gated Helix-mRNA weights
  (`MODEL_DIR/helix-mrna-hf/`).
- The codon usage table `elements.py` reads, the Codon Usage Database's Homo sapiens page saved as
  `WORK/kazusa-human/showcodon.html` (the address is in `elements.py`).
- `WORK/mrnabench-PMC12265608.xml`, the benchmark paper's full text from Europe PMC
  (`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12265608/fullTextXML`), for the published scores
  `final_tables.py` compares against.
