"""Recreate the three selected FDS submission CSVs from frozen full-train models."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone


ROOT = Path(__file__).resolve().parent
RECIPES = {
    "c09": {},
    "v334_highest_mean": {
        "CatBoost": ("F02", "F04", "N02", "N03", "N04"),
        "XGBoost": (),
        "LR_C10": ("F02", "N03"),
    },
    "v334_r_both": {
        "CatBoost": ("F02", "F04", "N02", "N03", "N04"),
        "XGBoost": ("F02", "N01", "N02", "N03"),
        "LR_C10": (),
    },
}
MODEL_FILES = {
    "CatBoost": "model-CatBoost.joblib",
    "XGBoost": "model-XGBoost.joblib",
    "LR_C10": "model-LR_C10.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(data_dir: Path):
    from src.refit_goal70_pm2_submission2_all13_v1 import read_jsonl

    train, y = read_jsonl(data_dir / "train.jsonl", training=True)
    test, _ = read_jsonl(data_dir / "test.jsonl")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    return train, y, test, sample


def feature_designs(train, test):
    from src.refit_goal70_c09_v332_submissions import _c09_matrices
    from src.goal70_c09_candidate_features_v332 import _recent_move_difference
    from src.goal70_user_feature_code_v330 import extract_user_features
    from src.projectbaseline_cycle190_status_move_streak import move_recency_features
    from src.refit_goal70_pm2_submission2_all13_v1 import statused_active_names

    train_ids = np.asarray([int(row["battle_id"]) for row in train], dtype=int)
    test_ids = np.asarray([int(row["battle_id"]) for row in test], dtype=int)
    train_base, lookup, conflicts, encoder = _c09_matrices(train, train_ids)
    test_base, _, _, _ = _c09_matrices(
        test, test_ids, lookup=lookup, conflicts=conflicts, encoder=encoder
    )
    key_map = {"CatBoost": "CatBoost", "XGBoost": "XGBoost",
               "LR_C10": "LogisticRegression_C10"}
    train_base = {name: train_base[key] for name, key in key_map.items()}
    test_base = {name: test_base[key] for name, key in key_map.items()}

    def candidate_frames(records):
        streaks = pd.DataFrame([move_recency_features(row)[0] for row in records])
        recent = pd.DataFrame({
            "move_made_diff_mean_recent10_context": [
                _recent_move_difference(row) for row in records
            ]
        })
        user = pd.DataFrame([extract_user_features(row) for row in records])
        status = statused_active_names(records)

        def max_run(values):
            best = current = 0
            for value in values:
                current = current + 1 if value else 0
                best = max(best, current)
            return best

        tail_rows = []
        for record in records:
            timeline = (record.get("battle_timeline") or [])[:30]
            if [turn.get("turn") for turn in timeline] != list(range(1, 31)):
                raise ValueError("Expected ordered timeline snapshots 1..30")
            row = {}
            for side in ("p1", "p2"):
                row[f"{side}_no_move_record_run_max_t21_30"] = max_run(
                    [turn.get(f"{side}_move_details") is None for turn in timeline[20:30]]
                )
            tail_rows.append(row)
        tail = pd.DataFrame(tail_rows)
        candidates = {
            "F02": streaks[["p2_trailing_status_move_turns"]],
            "F04": recent,
            "N01": status[["p2_unique_active_names_with_nonterminal_status"]],
            "N02": user[["encounter_hp_delta_p2_mean"]],
            "N03": user[["p2_lead_type_fighting"]],
            "N04": tail,
        }
        for token, frame in candidates.items():
            if not np.isfinite(frame.to_numpy(dtype=float)).all():
                raise ValueError(f"Non-finite feature values in {token}")
        return candidates

    train_candidates = candidate_frames(train)
    test_candidates = candidate_frames(test)
    train_designs = {"c09": train_base}
    test_designs = {"c09": test_base}
    for recipe_id in ("v334_highest_mean", "v334_r_both"):
        for base, candidates, designs in (
            (train_base, train_candidates, train_designs),
            (test_base, test_candidates, test_designs),
        ):
            current = {}
            for model_name, tokens in RECIPES[recipe_id].items():
                frame = base[model_name].copy()
                for token in tokens:
                    frame = frame.join(candidates[token])
                if frame.columns.duplicated().any() or not np.isfinite(
                    frame.to_numpy(dtype=float)
                ).all():
                    raise ValueError(f"Invalid feature design: {recipe_id}/{model_name}")
                current[model_name] = frame
            designs[recipe_id] = current
    return train_designs, test_designs, train_ids, test_ids


def positive_probability(model, matrix):
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Expected one positive class 1; found {classes.tolist()}")
    return model.predict_proba(matrix)[:, int(matches[0])]


def predict_recipe(recipe_id, train_designs, test_designs, y, test_ids, sample_ids,
                   output_dir, refit_models):
    model_dir = ROOT / "models" / recipe_id
    probabilities = {}
    model_columns = {}
    for name, filename in MODEL_FILES.items():
        bundle = joblib.load(model_dir / filename)
        expected_columns = bundle["columns"]
        model_columns[name] = expected_columns
        frame = test_designs[name]
        if frame.columns.duplicated().any() or set(frame.columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(expected_columns))
            raise ValueError(f"{recipe_id}/{name} schema mismatch; missing={missing}, extra={extra}")
        matrix = frame.loc[:, expected_columns]
        model = bundle["model"]
        if refit_models:
            train_frame = train_designs[name]
            if set(train_frame.columns) != set(expected_columns):
                raise ValueError(f"{recipe_id}/{name} training schema differs from artifact")
            model = clone(model)
            model.fit(train_frame.loc[:, expected_columns], y)
        probabilities[name] = positive_probability(model, matrix)

    blend = np.mean([probabilities[name] for name in MODEL_FILES], axis=0)
    prediction = (blend >= 0.5).astype(int)
    component = pd.DataFrame({
        "battle_id": test_ids,
        "catboost_probability": probabilities["CatBoost"],
        "xgboost_probability": probabilities["XGBoost"],
        "logistic_regression_probability": probabilities["LR_C10"],
        "blend_probability": blend,
        "player_won": prediction,
    }).set_index("battle_id").loc[sample_ids].reset_index()
    submission = component.loc[:, ["battle_id", "player_won"]]
    if submission.isna().any().any() or not submission["player_won"].isin([0, 1]).all():
        raise ValueError(f"Invalid output values for {recipe_id}")

    recipe_out = output_dir / recipe_id
    recipe_out.mkdir(parents=True, exist_ok=True)
    components_path = recipe_out / "component_predictions.csv"
    submission_path = recipe_out / "submission.csv"
    # CRLF matches the original Windows-generated CSV files byte-for-byte.
    component.to_csv(components_path, index=False, lineterminator="\r\n")
    submission.to_csv(submission_path, index=False, lineterminator="\r\n")

    expected_hash = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))[
        "recipes"][recipe_id]["submission_sha256"]
    actual_hash = sha256(submission_path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{recipe_id}: CSV bytes differ from frozen artifact; "
            f"expected {expected_hash}, got {actual_hash}"
        )
    metadata = {
        "recipe_id": recipe_id,
        "model_mode": "refit_from_train" if refit_models else "frozen_full_train_artifact",
        "model_family": "equal-weight soft-voting ensemble",
        "components": list(MODEL_FILES),
        "weights": {name: 1 / 3 for name in MODEL_FILES},
        "threshold": 0.5,
        "positive_class": 1,
        "feature_counts": {name: len(columns) for name, columns in model_columns.items()},
        "feature_columns": model_columns,
        "submission_rows": len(submission),
        "submission_sha256": actual_hash,
        "source_submission_sha256": expected_hash,
        "matches_source_submission_sha256": actual_hash == expected_hash,
        "model_artifact_sha256": {
            filename: sha256(model_dir / filename) for filename in MODEL_FILES.values()
        },
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "catboost", "joblib")
        },
    }
    (recipe_out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return submission_path, components_path, actual_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing the competition JSONL/CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--refit-models", action="store_true",
                        help="Retrain cloned estimators on all 10,000 labeled rows")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    required = [data_dir / name for name in
                ("train.jsonl", "test.jsonl", "sample_submission.csv")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError(f"Missing competition files in {data_dir}")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing or changed source file: {relative}")
    for recipe_id in RECIPES:
        for filename, expected in manifest["recipes"][recipe_id]["artifacts"].items():
            path = ROOT / "models" / recipe_id / filename
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"Missing or changed model: {recipe_id}/{filename}")

    train, y, test, sample = read_records(data_dir)
    train_ids = [int(row["battle_id"]) for row in train]
    test_ids = [int(row["battle_id"]) for row in test]
    if len(train) != 10000 or len(y) != 10000 or len(test) != 5000:
        raise ValueError(f"Unexpected sizes: train={len(train)}, test={len(test)}")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("battle_id must be unique in train and test")
    if list(sample.columns) != ["battle_id", "player_won"]:
        raise ValueError("Unexpected sample_submission schema")
    if sample["battle_id"].tolist() != test_ids:
        raise ValueError("sample_submission order does not match test.jsonl")
    if set(np.unique(y)) != {0, 1} or sample["battle_id"].duplicated().any():
        raise ValueError("Unexpected target classes or duplicate sample IDs")

    train_designs, test_designs, _, test_ids = feature_designs(train, test)
    results = []
    for recipe_id in RECIPES:
        path, components, digest = predict_recipe(
            recipe_id, train_designs[recipe_id], test_designs[recipe_id], y, test_ids,
            sample["battle_id"].to_numpy(), output_dir, args.refit_models
        )
        results.append({"recipe": recipe_id, "submission": str(path), "sha256": digest,
                        "refit_models": args.refit_models})
        print(json.dumps({"recipe": recipe_id, "submission": str(path),
                          "components": str(components), "sha256": digest}, indent=2))
    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_target_read": False,
        "input_sha256": {path.name: sha256(path) for path in required},
        "outputs": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
