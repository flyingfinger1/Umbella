# Umbella — Plant 2D→3D Bridge

Research project: from a single 2D plant photo, derive a plausible 3D structural hypothesis, and test whether this intermediate layer improves species identification on families where flat 2D classifiers struggle.

**Initial focus:** Apiaceae (umbellifers) — diagnostic features here are often structural and spatial (umbellet arrangement, branching angles) and hard to capture in flat photos. Safety-relevant: contains edible species side by side with highly toxic ones.

## Status

Early stage, but the **pipeline is closed end-to-end**:

```
SpeciesSpec  →  Skeleton  →  Point cloud  →  HPR (one-sided)  →  RGB / labels / depth
   ↑                                                                    ↓
   └──── value ranges calibrated                                   CNN classifier
         against ID keys                                       (synth test: 89–98 %)
                                                                       ↑
                                                          Pheno4D skeletons fed in
                                                          via the same data structure
```

Six calibrated Apiaceae species (hogweed, hemlock, wild carrot, cow parsley, fool's parsley, parsnip) plus Pheno4D integration (maize / tomato). **Two classifiers in a hybrid setup:**

- **Synth-CNN** (~843k params, several variants v6–v9): currently v9, with *online* augmentation (fresh background / shading / color jitter per training sample, drawn from a curated 88-image background pool of Pexels + Ideogram + Leonardo, all subject-free). Synth test 89.6 %.
- **Leaf-CNN** (ResNet-18 fine-tune on 1217 curated iNaturalist DACH images, with class weights against imbalance): 76.0 % test, all classes ≥60 % recall.

**Real-photo state (n=9 cow-parsley photos, statistically not meaningful):** no model variant reliably exceeds 1–2 Top-1 hits out of 9. An earlier hybrid (v7 + leaf, soft 30/70) reached 2/9 as an existence proof; not reproduced with v9. The most important open finding across three iterations (v6 / v7 / v9): **synth-test accuracy does not correlate with real-photo performance.** Before further architecture iterations a larger, pre-experimentally curated real-photo test set (order of 50+) is needed.

## Quickstart

```bash
git clone https://github.com/flyingfinger1/Umbella.git
cd Umbella
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy plotly scipy scikit-learn torch
```

### Synthetic pipeline (no dataset required)

```bash
.venv/Scripts/python.exe notebooks/09_calibrated_apiaceae.py     # 6 species, 3 instances each, separate HTML
.venv/Scripts/python.exe notebooks/14_render_synthetic.py        # render RGB / labels / depth
.venv/Scripts/python.exe notebooks/15_build_training_set.py      # build training set
.venv/Scripts/python.exe notebooks/17_train_classifier.py        # train the CNN
```

### Pheno4D integration

```bash
curl -L -o data/raw/Pheno4D.zip https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip
unzip data/raw/Pheno4D.zip -d data/
.venv/Scripts/python.exe notebooks/01_explore_pheno4d.py         # overview
.venv/Scripts/python.exe notebooks/04_skeleton.py Tomato03       # extract skeleton
.venv/Scripts/python.exe notebooks/06_build_skeleton_corpus.py   # serialize all 126 skeletons
```

Output goes to `notebooks/output/` as interactive HTML.

## Repository layout

```
src/
  datasets/pheno4d.py             Pheno4D point-cloud loader
  geometry/
    skeleton.py                   geodesic polylines + Steiner-tree skeletons
    pointcloud.py                 cylinder surface sampling along the skeleton
    visibility.py                 Hidden Point Removal (Katz et al. 2007)
    render.py                     perspective 2D projection
    augment.py                    background / shading / color jitter
  synthetic/
    apiaceae.py                   L-system-style compound umbel generator
    species.py                    calibrated value ranges per species
  eval/features.py                27 skeleton-based structural features
  training/dataset.py             training-triplet generator (RGB / labels / depth)
  models/classifier.py            compact synth-CNN (~843k params)
  inference/
    predict.py                    single-image inference, synth model
    predict_leaf.py               single-image inference, leaf model (ResNet-18)
    hybrid.py                     synth + leaf ensemble (soft 30/70 default)
  leaf/                           real-photo classifier
    fetch_inaturalist.py          iNaturalist downloader (DACH + research-grade)
    dataset.py                    loader + observation-stratified split
    model.py                      ResNet-18 fine-tune
notebooks/
  01–04                           Pheno4D exploration + skeletonization
  05                              skeleton diagnostics
  06                              serialize skeleton corpus
  07                              maize vs. tomato classifier
  08–09                           generate synthetic Apiaceae
  10                              6-species classifier (structural features)
  11                              point clouds from skeletons
  12                              domain-gap analysis
  13                              Hidden Point Removal
  14                              2D render
  15–16                           build training set + QA
  17                              train CNN
  18                              overnight orchestrator (cosine LR + 768 px)
  19                              training on v5 (with Conium stem speckles)
  20                              v6 build + training (augmentation: BG / shading / jitter)
  21                              single-image inference helper
  22                              v7 build + training (botanical corrections + Pastinaca yellow)
  23                              real-photo classifier training (after iNat data review)
  24                              browser review tool for iNaturalist images
  25                              v8 build + training (synth + 50% iNat backgrounds)
  26                              hybrid inference eval (synth + leaf ensemble strategies)
  27                              v9 build (clean RGB) + training (online aug, curated BG pool)
  28                              v9 vs. v7 / v8 / leaf / hybrid on real-photo set
data/                             not in repo (see .gitignore)
```

## Background & methodology

- [`CLAUDE.md`](CLAUDE.md) — tech stack, conventions, directory structure, design questions (in German)
- [`research.md`](research.md) — background, state of the art, dataset survey, test-set methodology, **work diary** (chronological log of every substantial step, with rationale) (in German)

## Datasets

| Name | Content | License | Size |
|---|---|---|---|
| Pheno4D | 7 maize + 7 tomato, point clouds across 2–3 weeks, organ-labeled | CC BY | 4.44 GB |
| ROSE-X | 11 rose bushes, X-ray CT, voxel + point clouds | CC BY 4.0 | 1.53 GB |
| synthetic (`data/training/v9/`) | 200 instances × 6 Apiaceae × 4 views = 4800 *clean* RGB + label + depth triplets (augmentation runs online in the dataloader) | generated | ~80 MB |
| background pool (`data/bg_textures/`) | 88 subject-free images (64 Pexels outdoor textures + 13 indoor + 11 outdoor scenes from Ideogram / Leonardo, manually curated) | mixed CC0 / AI-generated | ~30 MB |
| iNat-Apiaceae (`data/leaf_images/`) | 1217 curated real photos, DACH / research-grade, 6 species | iNaturalist | ~150 MB |

Apiaceae-specific 3D data does not exist publicly — it is generated synthetically (L-system calibrated against ID keys) and, eventually, through the project's own captures. Details in [`research.md`](research.md).
