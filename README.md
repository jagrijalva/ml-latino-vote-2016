# ml-latino-vote-2016

Inferential Feature Analysis of Latino Trump support using the 2016 Collaborative Multiracial Post-Election Survey (CMPS).

**DV:** Binary indicator of Latino vote for Donald Trump.

**Method:** Random Forest (`ranger`) + SHAP values (`treeshap` / `shapviz`), four-tier progressive exclusion framework, bootstrap rank-stability analysis.

## Key files

- `analysis.qmd` — end-to-end analysis pipeline. Runs data cleaning, imputation, RF models (Tiers 1–4), SHAP decomposition, and bootstrap. Saves fitted objects to `data/derived/ifa_results_2016.rds` and renders `analysis.pdf`.
- `analysis.pdf` — rendered output with manuscript-ready figures and tables.
- `ml-latino-vote-2016.Rproj` — RStudio project file.
- `docs/` — CMPS 2016 codebook, questionnaire, and working codebook notes.

## Folder structure

```
analysis.qmd        # analysis + reporting pipeline
analysis.pdf        # rendered output
docs/               # codebook, questionnaire, feature notes
data/
  raw/              # ICPSR raw data (gitignored; download required)
  derived/          # ifa_results_2016.rds (gitignored; regenerated on render)
```

## Reproducing

1. Obtain the 2016 CMPS raw data from ICPSR (study 38040) and place `38040-0001-Data.rda` under `data/raw/`.
2. Open `ml-latino-vote-2016.Rproj` in RStudio.
3. Render `analysis.qmd`. This fits all models and writes `data/derived/ifa_results_2016.rds`.

## Pooled-paper contract

The pooled 2016/2020/2024 paper consumes `data/derived/ifa_results_2016.rds`. The file name and location are fixed; the object bundles metadata, seeds, model matrices, SHAP output, bootstrap ranks, and AUC summaries.

## What is gitignored

- `data/` — license-restricted ICPSR raw data and regenerated derived objects
- `*.rds` — model bundles (regenerated on render)
- `scratch/` — temporary working space
- Quarto render artifacts (`*_files/`, `.quarto/`, `*.tex`, `*.html`, etc.)

## Related repositories

- [`ml-latino-vote-2020`](https://github.com/jagrijalva/ml-latino-vote-2020) — parallel analysis on the 2020 CMPS.
- [`ml-latino-vote-2024`](https://github.com/jagrijalva/ml-latino-vote-2024) — parallel analysis on the 2024 CMPS.

## License

MIT — see [`LICENSE`](LICENSE).
