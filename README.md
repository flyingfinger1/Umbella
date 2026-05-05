# Umbella — Plant 2D→3D Bridge

Forschungsprojekt: aus einzelnen 2D-Pflanzenfotos plausible 3D-Strukturhypothesen ableiten und prüfen, ob diese Zwischenebene die Artbestimmung bei strukturell schwer unterscheidbaren Familien verbessert.

**Initialer Fokus:** Apiaceae (Doldenblütler) — diagnostische Merkmale sind dort oft strukturell-räumlich (Anordnung der Döldchen, Verzweigungswinkel) und auf flachen Fotos schwer fassbar. Sicherheitsrelevant: enthält essbare neben hochgiftigen Arten.

## Status

Frühe Phase, aber **Pipeline ist beidseitig geschlossen**:

```
SpeciesSpec  →  Skeleton  →  Point cloud  →  HPR (one-sided)  →  RGB / labels / depth
   ↑                                                                    ↓
   └──── kalibrierte Wertebereiche                                  CNN-Klassifikator
         aus Bestimmungsschlüsseln                              (Synth-Test: 89–98 %)
                                                                       ↑
                                                          Pheno4D-Skelette einseitbar
                                                          über dieselbe Datenstruktur
```

Aktuell sechs kalibrierte Apiaceae-Arten (Wiesen-Bärenklau, Schierling, Wilde Möhre, Wiesen-Kerbel, Hundspetersilie, Pastinak) plus Pheno4D-Anbindung (Mais/Tomate). **Zwei Klassifikatoren in Hybrid-Setup:**

- **Synth-CNN** (~843k Params, mehrere Varianten v6–v9): aktuell v9 mit *online*-Augmentation (frische BG/Shading/Color-Jitter pro Trainings-Sample, kuratierter 88-Bilder-Hintergrund-Pool aus Pexels + Ideogram + Leonardo, alle subjekt-frei). Synth-Test 89.6 %.
- **Leaf-CNN** (ResNet-18 fine-tune auf 1217 kuratierten iNaturalist-DACH-Bildern, mit Class-Weights gegen Imbalance): 76.0 % Test, alle Klassen ≥60 % Recall.

**Real-Foto-Stand (n=9 Wiesen-Kerbel-Fotos, statistisch nicht aussagekräftig):** keine Modellvariante schlägt zuverlässig 1–2 Top-1-Treffer auf 9. Frühere Hybrid-Variante (v7+Leaf Soft 30/70) erreichte 2/9 als Existenzbeweis, mit v9 nicht reproduziert. Wichtigster offener Befund über drei Iterationen (v6/v7/v9): **Synth-Test-Accuracy korreliert nicht mit Real-Foto-Performance**. Vor weiteren Architektur-Iterationen wird ein größeres, prä-experimentell kuratiertes Real-Foto-Test-Set (Größenordnung 50+) gebraucht.

## Quickstart

```bash
git clone https://github.com/flyingfinger1/Umbella.git
cd Umbella
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy plotly scipy scikit-learn torch
```

### Synthetische Pipeline (kein Datensatz nötig)

```bash
.venv/Scripts/python.exe notebooks/09_calibrated_apiaceae.py     # 6 Arten, 3 Instanzen, je eigenes HTML
.venv/Scripts/python.exe notebooks/14_render_synthetic.py        # rendert RGB / labels / depth
.venv/Scripts/python.exe notebooks/15_build_training_set.py      # baut Trainingsset
.venv/Scripts/python.exe notebooks/17_train_classifier.py        # trainiert das CNN
```

### Pheno4D-Anbindung

```bash
curl -L -o data/raw/Pheno4D.zip https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip
unzip data/raw/Pheno4D.zip -d data/
.venv/Scripts/python.exe notebooks/01_explore_pheno4d.py         # Übersicht
.venv/Scripts/python.exe notebooks/04_skeleton.py Tomato03       # Skelett extrahieren
.venv/Scripts/python.exe notebooks/06_build_skeleton_corpus.py   # alle 126 Skelette serialisieren
```

Output landet als interaktive HTML in `notebooks/output/`.

## Repository-Struktur

```
src/
  datasets/pheno4d.py             Loader für Pheno4D-Punktwolken
  geometry/
    skeleton.py                   geodätische Polylines + Steiner-Tree-Skelette
    pointcloud.py                 Zylinder-Surface-Sampling am Skelett
    visibility.py                 Hidden Point Removal (Katz et al. 2007)
    render.py                     perspektivische 2D-Projektion
    augment.py                    Background / Shading / Color-Jitter
  synthetic/
    apiaceae.py                   L-System-artiger Doppeldolden-Generator
    species.py                    kalibrierte Wertebereiche pro Art
  eval/features.py                27 Skelett-basierte Strukturmerkmale
  training/dataset.py             Trainings-Triplet-Generator (RGB / Labels / Depth)
  models/classifier.py            kompakter Synth-CNN (~843k Params)
  inference/
    predict.py                    Single-Image-Inferenz Synth-Modell
    predict_leaf.py               Single-Image-Inferenz Leaf-Modell (ResNet-18)
    hybrid.py                     Ensemble Synth + Leaf (Soft 30/70 default)
  leaf/                           Real-Photo-Klassifikator (Ansatz D)
    fetch_inaturalist.py          iNaturalist-Downloader (DACH + research-grade)
    dataset.py                    Loader + observation-stratified Split
    model.py                      ResNet-18 Fine-tune
notebooks/
  01–04                           Pheno4D-Exploration + Skelettierung
  05                              Skelett-Diagnostik
  06                              Skelett-Korpus serialisieren
  07                              Mais vs. Tomate Klassifikator
  08–09                           synthetische Apiaceae generieren
  10                              6-Arten-Klassifikator (Strukturmerkmale)
  11                              Punktwolken aus Skeletten
  12                              Domain-Gap-Analyse
  13                              Hidden-Point-Removal
  14                              2D-Render
  15–16                           Trainings-Datensatz bauen + QA
  17                              CNN trainieren
  18                              Overnight-Orchestrator (cosine LR + 768er-Auflösung)
  19                              Training auf v5 (mit Conium-Stamm-Speckles)
  20                              v6-Build + Training (Augmentation: BG / Shading / Jitter)
  21                              Single-image inference helper
  22                              v7-Build + Training (botanical corrections + Pastinaca yellow)
  23                              Real-photo-classifier training (after iNat data review)
  24                              Browser review tool for iNaturalist images
  25                              v8-Build + Training (synth + 50% iNat backgrounds)
  26                              Hybrid-Inferenz Eval (Synth + Leaf Ensemble-Strategien)
  27                              v9-Build (clean RGB) + Training (online aug, curated BG-Pool)
  28                              v9 vs v7/v8/leaf/hybrid auf Real-Foto-Set
data/                             nicht im Repo (siehe .gitignore)
```

## Hintergrund & Methodik

- [`CLAUDE.md`](CLAUDE.md) — Tech-Stack, Konventionen, Verzeichnisstruktur, Designfragen
- [`research.md`](research.md) — Hintergrund, State of the Art, Datensatz-Recherche, Test-Set-Methodik, **Arbeitstagebuch** (chronologisch jeder substantielle Schritt mit Begründung)

## Datensätze

| Name | Inhalt | Lizenz | Größe |
|---|---|---|---|
| Pheno4D | 7 Mais + 7 Tomate, Punktwolken über 2–3 Wochen, organ-gelabelt | CC BY | 4.44 GB |
| ROSE-X | 11 Rosenbüsche, X-ray-CT, Voxel + Punktwolken | CC BY 4.0 | 1.53 GB |
| synthetisch (`data/training/v9/`) | 200 Instanzen × 6 Apiaceae × 4 Views = 4800 *clean* RGB+Label+Depth-Triplets (Augmentation läuft online im Dataloader) | erzeugt | ~80 MB |
| Hintergrund-Pool (`data/bg_textures/`) | 88 subjekt-freie Bilder (64 Pexels-Outdoor-Texturen + 13 Indoor- + 11 Outdoor-Szenen aus Ideogram/Leonardo, manuell kuratiert) | gemischt CC0 / KI-erzeugt | ~30 MB |
| iNat-Apiaceae (`data/leaf_images/`) | 1217 kuratierte Real-Fotos, DACH/research-grade, 6 Arten | iNaturalist | ~150 MB |

Apiaceae-spezifische 3D-Daten existieren nicht öffentlich — werden synthetisch (L-System gegen Bestimmungsschlüssel kalibriert) und perspektivisch eines Tages durch Eigenaufnahmen erzeugt. Details in [`research.md`](research.md).
