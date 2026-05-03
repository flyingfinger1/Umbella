# Plant 2D→3D Bridge

Forschungs- und Entwicklungsprojekt: Aus einzelnen 2D-Pflanzenfotos plausible 3D-Strukturhypothesen ableiten, mit dem Ziel, die Artbestimmung bei strukturell schwer unterscheidbaren Kladen zu verbessern.

## Ziel

Eine Pipeline und Modellfamilie entwickeln, die aus einem Smartphone-Foto einer Pflanze eine Strukturhypothese erzeugt (Punktwolke, Skelettgraph oder prozedurale Modellparameter), und prüfen, ob diese Zwischenebene die Bestimmungsgenauigkeit gegenüber reinem 2D-CNN-Ansatz messbar erhöht.

**Initialer Fokus:** Apiaceae (Doldenblütler). Begründung in `RESEARCH.md`.

## Aktueller Stand

Frühe Konzeptphase. Noch kein Code, noch keine Datenauswahl getroffen.

## Geplanter Tech-Stack

- **ML-Framework:** PyTorch (Standard im 3D-Vision-Bereich, beste Modellverfügbarkeit)
- **3D-Verarbeitung:** Open3D, PyTorch3D, ggf. nerfstudio / gsplat für NeRF/Gaussian Splatting
- **Synthetische Daten:** GroIMP oder eigenes L-System (Python) für prozedurale Apiaceae-Geometrien
- **Datenverwaltung:** DVC oder schlicht git-lfs, je nach Datengröße
- **Experiment-Tracking:** Weights & Biases oder MLflow self-hosted
- **Deployment-Idee (später):** Docker, ggf. eigener VPS

Sprache primär Python. Keine Java-Komponenten geplant.

## Datensätze (Kandidaten, noch nicht final)

Details und Lizenzfragen in `RESEARCH.md`.

- Pheno4D (Mais/Tomate, zeitliche Punktwolken)
- ROSE-X (Rosenbüsche)
- Plant3D
- Eigene synthetische Apiaceae-Daten via L-System-Simulation
- Ggf. Eigenaufnahmen mit iPhone-LiDAR / Polycam für Validierung im Freiland

## Repository-Konventionen

- **Branches:** `main` stabil, Feature-Branches als `feat/<thema>`, Experimente als `exp/<id>-<kurzname>`
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `exp:`)
- **Code-Style:** ruff + black für Python, type hints wo sinnvoll
- **Tests:** pytest, mindestens für Datenpipelines und Geometrie-Utilities
- **Notebooks:** unter `notebooks/`, mit `nbstripout` als pre-commit-Hook gegen Output-Diff-Müll

## Verzeichnisstruktur (geplant)

```
.
├── CLAUDE.md              # diese Datei
├── RESEARCH.md            # Hintergrund, State of the Art, Argumentation
├── README.md              # öffentlich, kurz
├── data/                  # nicht committed, via DVC/lfs
├── src/
│   ├── datasets/          # Loader für Pheno4D etc.
│   ├── models/            # Architekturen
│   ├── synthetic/         # L-System-Generator
│   ├── geometry/          # Skelett, Punktwolken-Utils
│   └── eval/              # Metriken, Vergleich gegen 2D-Baseline
├── notebooks/             # Explorationen
├── experiments/           # Konfigs (yaml/hydra)
└── tests/
```

## Offene Designfragen

Stand jetzt offen, sollen mit fortschreitender Recherche entschieden werden:

1. Repräsentation der 3D-Hypothese: Punktwolke vs. Skelettgraph vs. prozedurale Parameter vs. Gaussian Splat. Vermutlich Skelettgraph als Pivot, weil er sowohl strukturell interpretierbar als auch botanisch sinnvoll ist.
2. Wie viel synthetische Daten vs. echte Daten? Domain Gap ist bei Pflanzen erheblich.
3. Architektur: Encoder-Decoder mit expliziter 3D-Zwischenrepräsentation, oder implicit-field-Ansatz, oder Gaussian-Splatting-Vorhersage in einem Schritt.
4. Wie wird die "Verbesserung gegenüber 2D-Baseline" sauber gemessen? Braucht einen Test-Set mit echten Apiaceae-Bestimmungen, idealerweise mit Ground-Truth von Botaniker:innen.

## Arbeitsweise mit Claude Code

- Bei Architektur- oder Datensatzentscheidungen: erst `RESEARCH.md` lesen, dann gezielt fragen.
- Code-Änderungen mit kurzer Begründung im Commit, nicht nur "update".
- Experimente immer in `experiments/` als Config ablegen, nicht als Magic-Numbers im Code.
- Bei neuen Datensätzen: Lizenz und Zitationspflicht prüfen und in `RESEARCH.md` eintragen.
