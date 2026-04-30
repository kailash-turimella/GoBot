# Training Data

## Source

All game data comes from **CGOS — the Computer Go Server**, a dedicated 9×9 computer Go server with game records going back to 2008.

**Download archives:** http://www.yss-aya.com/cgos/

The monthly archives are listed under the 9×9 section. Each archive is a `.tar.bz2` file containing SGF game records organised by `year/month/day/`.

## What we used

~1.1 million 9×9 games across multiple years, downloaded as monthly `.tar.bz2` archives. The supervised training pipeline used 500,000 of these games in chunks of 40,000.

## Setup

1. Download the monthly archives from the CGOS link above
2. Place the `.tar.bz2` files in `AI/training/data/Games/`
3. Extract them:
   ```bash
   cd AI/training/data/Games
   for f in *.tar.bz2; do tar -xjf "$f"; done
   ```
4. Run supervised training:
   ```bash
   python -m AI.training.supervised --games AI/training/data/Games/ --max-games 50000 --epochs 20 --device mps --promote
   ```

## Samples

This folder contains 10 example SGF files so you can inspect the format without downloading the full dataset.

Each file is a standard SGF record. The training parser handles:
- Board size validation (`SZ[9]` only)
- Result extraction (`RE[B+...]` / `RE[W+...]`, including resigns)
- Move replay with full Go rules to verify legality
- Automatic skipping of malformed or unknown-result games
