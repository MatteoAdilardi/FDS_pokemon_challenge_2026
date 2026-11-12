# FDS Kaggle submission reproducer

This folder is a self-contained public repository payload for reproducing the three selected FDS competition submissions. It contains the feature-extraction source modules, the full-training fitted model artifacts, a runner, and a Kaggle notebook that clones this repository and writes one CSV per recipe.

## Selected recipes

All three are equal-weight soft-voting blends of CatBoost, XGBoost, and Logistic Regression. `battle_id` is used only to align output rows and never enters a predictor. Each output has the competition schema `battle_id,player_won`, 5,000 rows, and threshold 0.5.

| Recipe | CatBoost additions | XGBoost additions | Logistic Regression additions | Reported public score | Model feature counts (Cat / XGB / LR) | Expected CSV SHA-256 |
|---|---|---|---|---:|---:|---|
| `c09` | None beyond C09 | None beyond C09 | None beyond C09 | 0.69333 | 80 / 142 / 143 | `4d460eec5d89b503b0133c8b715847bd38d6e33d4602450c91cffcc2c3dfc1a1` |
| `v334_highest_mean` | F02, F04, N02, N03, N04 | None | F02, N03 | 0.69266 | 86 / 142 / 145 | `327817d8ef0e8b14456578a1de8f92e077d98231747f8200ad632422abf6bcd4` |
| `v337_cat_f06_pair` | F06 P1+P2 | None | None | 0.69800 | 82 / 142 / 143 | `60965035e9a44563a12c475e0b75db69c6849af0d9c978b8fec41376b982fc66` |

The three CSVs were matched byte-for-byte to the files supplied by the user. Their reported private scores are C09 0.65914, highest mean 0.66457, and V337 0.66285. The final competition ranking uses the private leaderboard. Scores are user-reported, not independently verified on Kaggle. V337 replaces the previously packaged R_BOTH recipe.

F06 adds only `p1_new_faint_names_t21_30` and `p2_new_faint_names_t21_30` to CatBoost C09. For each side, scan snapshots 1–30 and record the first terminal observation for each normalized active name (`status == "fnt"` or finite `hp_pct <= 0`). Count names whose first such observation falls in turns 21–30 inclusive. Repeated terminal snapshots count once; a name first terminal before turn 21 is excluded. These are observed names, not inferred hidden KOs or guaranteed individual identities. XGBoost and LR remain C09.

## What the notebook does

1. Clones the configured public GitHub repository: https://github.com/MatteoAdilardi/FDS_pokemon_challenge_2026.
2. Finds the attached competition files in `/kaggle/input`.
3. Installs the pinned Python packages listed in `requirements.txt`.
4. Rebuilds the exact model-specific feature matrices from the 10,000 labeled training rows and the first 30 visible timeline snapshots of each test battle.
5. Loads the saved estimators, predicts the test rows, and writes `submission.csv`, component probabilities, and metadata for each recipe under `/kaggle/working/submissions/`.
6. Checks each output against its recorded SHA-256 and fails if the CSV bytes differ from the selected source file.

The default mode loads the models already refitted on all 10,000 labeled rows. The optional `--refit-models` flag clones those estimators and trains them again on all 10,000 rows before prediction. The frozen-artifact mode is the recommended mode when the goal is to reproduce the exact selected CSV bytes; the refit mode is provided to make the training step executable and can differ across Python/library environments. Both modes stop with an error if the generated CSV hash differs from the selected source submission.

The script reads the test records only to create features and predictions. It rejects a `player_won` field in test records, never uses test labels, and writes predictions in `sample_submission.csv` order. The source modules needed for the original feature recipes are included under `src/` and `personal_model_2/`. The PM2 helper computes only the six columns consumed by C09; its values were checked against the historical extractor on 60 training and 60 test rows. Unused experiment-time type feature code is excluded.

## Run locally

Python 3.12 or newer is required. The verified local environment uses Python 3.14.8. A GPU is not required.

From this folder, with the challenge files in `../data/`:

```bash
python -m pip install -r requirements.txt
python run_submissions.py --data-dir ../data --output-dir outputs
```

To refit the three ensembles on all labels instead of loading the frozen estimators:

```bash
python run_submissions.py --data-dir ../data --output-dir outputs --refit-models
```

## Kaggle setup

Publish the contents of this folder at the root of https://github.com/MatteoAdilardi/FDS_pokemon_challenge_2026. The notebook already sets `REPOSITORY_URL` to its HTTPS clone URL. Attach the FDS competition dataset in Kaggle. The notebook needs Internet access to clone GitHub and install pinned packages; if the competition's final execution environment has Internet disabled, configure Kaggle's Dependency Manager before committing the notebook and use its offline package installation workflow.

After running all cells, download each CSV separately from `/kaggle/working/submissions/{c09,v334_highest_mean,v337_cat_f06_pair}/submission.csv`. Do not submit `component_predictions.csv`.

## Generated outputs

`outputs/` is the local default output directory for prediction with the saved full-training estimators. `outputs_refit/` is the directory used during verification with `--refit-models`. These are generated files, not runtime dependencies; both directories are excluded by `.gitignore` and are not included in this repository payload.

Each recipe writes `submission.csv`, `component_predictions.csv`, and `metadata.json`. The run also writes `run_manifest.json`. Refit models are fitted in memory and do not overwrite the saved model artifacts. The submission labels matched exactly in both local verification runs; component probabilities can differ slightly due to numerical rounding.

`.gitattributes` preserves LF endings for source files and treats model files as binary. The runner verifies the source and model hashes before building features. Submission CSV files use explicit CRLF endings to match the original Windows CSV bytes.

Source files are stored with LF line endings before their manifest hashes are computed. After updating this repository, refresh an existing Kaggle clone with `git -C /kaggle/working/fds_selected_submissions pull --ff-only` before rerunning the pipeline. The setup cell skips cloning when that directory already exists.

## Provenance and scope

`manifest.json` records source bundle names, the user-selected score labels, feature definitions, all model feature columns and pipeline parameters, model artifacts, and expected output hashes. Model artifacts were fitted on the complete 10,000-row labeled training set. No leaderboard labels or test targets are included. The notebook does not upload submissions.

Legacy feature names and formulas are retained for exact reproduction. In particular, the historical PM2 `p1_status_*` timing columns use P2 snapshots, and the `p2_status_*` timing columns use P1 snapshots. These names are not corrected during reproduction.
