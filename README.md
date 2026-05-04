# Umbella — Plant 2D→3D Bridge

Forschungsprojekt: aus einzelnen 2D-Pflanzenfotos plausible 3D-Strukturhypothesen ableiten und prüfen, ob diese Zwischenebene die Artbestimmung bei strukturell schwer unterscheidbaren Familien verbessert.

**Initialer Fokus:** Apiaceae (Doldenblütler) — diagnostische Merkmale sind dort oft strukturell-räumlich (Anordnung der Döldchen, Verzweigungswinkel) und auf flachen Fotos schwer fassbar. Sicherheitsrelevant: enthält essbare neben hochgiftigen Arten.

## Status

Frühe Phase, aber **Pipeline ist beidseitig geschlossen**:

```
SpeciesSpec  →  Skeleton  →  Point cloud  →  HPR (one-sided)  →  RGB / labels / depth
   ↑                                                                    ↓
   └──── kalibrierte Wertebereiche                                  CNN-Klassifikator
         aus Bestimmungsschlüsseln                              (~89 % Test-Accuracy)
                                                                       ↑
                                                          Pheno4D-Skelette einseitbar
                                                          über dieselbe Datenstruktur
```

Aktuell sechs kalibrierte Apiaceae-Arten (Wiesen-Bärenklau, Schierling, Wilde Möhre, Wiesen-Kerbel, Hundspetersilie, Pastinak) plus Pheno4D-Anbindung (Mais/Tomate). Ein erstes CNN klassifiziert sie aus synthetisch gerenderten Bildern mit ~89 % Test-Accuracy; verbleibende Confusion liegt fast vollständig auf dem Conium ↔ Anthriscus-Paar (Stand vor Stamm-Speckle-Iteration).

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
  synthetic/
    apiaceae.py                   L-System-artiger Doppeldolden-Generator
    species.py                    kalibrierte Wertebereiche pro Art
  eval/features.py                27 Skelett-basierte Strukturmerkmale
  training/dataset.py             Trainings-Triplet-Generator (RGB / Labels / Depth)
  models/classifier.py            kompakter CNN (~843k Params)
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
  18                              Overnight-Orchestrator
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
| synthetisch (`data/training/v3/`) | 200 Instanzen × 6 Apiaceae × 4 Views = 4800 RGB+Label+Depth-Triplets | erzeugt | ~310 MB |

Apiaceae-spezifische 3D-Daten existieren nicht öffentlich — werden synthetisch (L-System gegen Bestimmungsschlüssel kalibriert) und perspektivisch eines Tages durch Eigenaufnahmen erzeugt. Details in [`research.md`](research.md).
