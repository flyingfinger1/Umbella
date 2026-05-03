# Umbella — Plant 2D→3D Bridge

Forschungsprojekt: aus einzelnen 2D-Pflanzenfotos plausible 3D-Strukturhypothesen ableiten und prüfen, ob diese Zwischenebene die Artbestimmung bei strukturell schwer unterscheidbaren Familien verbessert.

**Initialer Fokus:** Apiaceae (Doldenblütler) — diagnostische Merkmale sind dort oft strukturell-räumlich (Anordnung der Döldchen, Verzweigungswinkel) und auf flachen Fotos schwer fassbar. Sicherheitsrelevant: enthält essbare neben hochgiftigen Arten.

**Status:** frühe Konzept- und Explorationsphase.

## Quickstart

```bash
git clone https://github.com/flyingfinger1/Umbella.git
cd Umbella
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy plotly scipy
```

Pheno4D-Datensatz herunterladen (4.4 GB, CC BY) und nach `data/Pheno4D/` entpacken:

```bash
curl -L -o data/raw/Pheno4D.zip https://www.ipb.uni-bonn.de/html/projects/Pheno4D/Pheno4D.zip
unzip data/raw/Pheno4D.zip -d data/
```

Erste Visualisierung:

```bash
.venv/Scripts/python.exe notebooks/01_explore_pheno4d.py
```

Output landet als interaktive HTML in `notebooks/output/`.

## Repository-Struktur

```
src/
  datasets/pheno4d.py      Loader für Pheno4D-Punktwolken
  geometry/skeleton.py     Skelettextraktion (geodätisch + Steiner-Tree)
notebooks/
  01_explore_pheno4d.py    Datensatz-Übersicht, eine Wolke laden
  02_growth_timeseries.py  Wachstumsanimation einer Pflanze
  03_leaf_instances.py     Per-Blatt-Isolierung + Stats
  04_skeleton.py           Vollständige Pflanzen-Skelettierung
data/                      nicht im Repo (siehe .gitignore)
```

## Hintergrund & Methodik

- [`CLAUDE.md`](CLAUDE.md) — Tech-Stack, Konventionen, Verzeichnisstruktur, Designfragen
- [`research.md`](research.md) — Hintergrund, State of the Art, Datensatz-Recherche, Test-Set-Methodik, Arbeitstagebuch

## Datensätze

| Name | Inhalt | Lizenz | Größe |
|---|---|---|---|
| Pheno4D | 7 Mais + 7 Tomate, Punktwolken über 2–3 Wochen, organ-gelabelt | CC BY | 4.44 GB |
| ROSE-X | 11 Rosenbüsche, X-ray-CT, Voxel + Punktwolken | CC BY 4.0 | 1.53 GB |

Apiaceae-spezifische 3D-Daten existieren nicht öffentlich — werden synthetisch (L-System) und/oder durch Eigenaufnahmen erzeugt. Details in [`research.md`](research.md).
