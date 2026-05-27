#!/usr/bin/env python
"""
OFETNow-ML generic prediction script.

Supports JSON and CSV input. JSON is recommended.
If CSV input is provided, it is automatically converted to JSON before inference.

Default usage from repository root:
    python examples/run_prediction.py

Custom input:
    python examples/run_prediction.py --input data/future_blind.csv

Custom paths:
    python examples/run_prediction.py --input data/example_input.json --model-dir models --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors

warnings.filterwarnings("ignore")


SMILES_COLUMN = "SMILES"

ROBUST_FEATURES = ["N_AVE", "Capacitance", "Aspect_Ratio", "Annealing_Temperature"]
MINMAX_FEATURES = ["PDI"]
CATEGORICAL_FEATURES = [
    "GATE",
    "Dielectric",
    "Fabrication_Method",
    "Solvent_of_Spin_Coating",
    "Test_Environment",
    "Type",
    "S/D",
]

MORGAN_RADIUS = 8
MORGAN_N_BITS = 2048

SELECTED_DESCRIPTOR_NAMES_FALLBACK = ['SMILES', 'MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex', 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA8', 'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 'HeavyAtomCount', 'NOCount', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAmideBonds', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'Phi', 'RingCount', 'MolLogP', 'MolMR', 'fr_Ar_N', 'fr_C_O', 'fr_C_O_noCOO', 'fr_NH0', 'fr_amide', 'fr_aniline', 'fr_aryl_methyl', 'fr_benzene', 'fr_bicyclic', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_halogen', 'fr_imide', 'fr_nitrile', 'fr_para_hydroxylation', 'fr_pyridine', 'fr_sulfide', 'fr_thiophene', 'fr_unbrch_alkane']


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run OFETNow-ML prediction on JSON or CSV input."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=repo_root_default,
        help="Repository root. Default: parent folder of examples/.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input .json or .csv file. Default: <project-root>/data/example_input.json.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing released model files. Default: <project-root>/models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for prediction outputs. Default: <project-root>/outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix. Default: input file stem.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def read_feature_list(path: Path) -> list[str]:
    path = require_file(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dataframe_to_json_records(df: pd.DataFrame, json_path: Path) -> Path:
    records = df.where(pd.notna(df), None).to_dict(orient="records")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return json_path


def load_prediction_input(
    input_path: Path, output_dir: Path
) -> tuple[pd.DataFrame, Path]:
    input_path = require_file(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data), input_path

    if suffix == ".csv":
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding="ISO-8859-1")
        json_path = output_dir / input_path.with_suffix(".json").name
        dataframe_to_json_records(df, json_path)
        print(f"CSV converted to JSON: {json_path}")
        return df, json_path

    raise ValueError("Input file must be .json or .csv")


def clean_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\r", "", regex=False)
            .str.replace("\n", "", regex=False)
            .str.strip()
        )
        df.loc[df[col].isin(["nan", "None", "NaN"]), col] = ""

    required = [
        SMILES_COLUMN,
        "PDI",
        "Aspect_Ratio",
        "Capacitance",
        "GATE",
        "Dielectric",
        "Fabrication_Method",
        "Solvent_of_Spin_Coating",
        "Test_Environment",
        "Type",
        "S/D",
        "Annealing_Temperature",
    ]
    missing_required = [c for c in required if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required input columns: {missing_required}")

    for col in [
        "Mn_Kda",
        "N_AVE",
        "PDI",
        "Aspect_Ratio",
        "Capacitance",
        "Annealing_Temperature",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Mn_Kda" not in df.columns:
        df["Mn_Kda"] = np.nan

    if "N_AVE" not in df.columns:
        df["N_AVE"] = np.nan

    missing_nave = df["N_AVE"].isna()
    if missing_nave.any():
        if df["Mn_Kda"].isna().all():
            print(
                f"Warning: {missing_nave.sum()} entries have no N_AVE and no Mn_Kda. "
                "N_AVE will remain NaN for these entries."
            )
        else:
            print(
                f"Calculating N_AVE for {missing_nave.sum()} entries "
                "from Mn_Kda and monomer molecular weight."
            )

        def monomer_mw(smi: str) -> float:
            mol = Chem.MolFromSmiles(str(smi))
            return Descriptors.MolWt(mol) if mol is not None else np.nan

        monomer_mw_values = df.loc[missing_nave, SMILES_COLUMN].apply(monomer_mw)
        df.loc[missing_nave, "N_AVE"] = (
            df.loc[missing_nave, "Mn_Kda"] * 1000 / monomer_mw_values
        )

    return df


def smiles_to_morgan(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros((MORGAN_N_BITS,), dtype=np.uint8)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_N_BITS,
    )
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    for i in range(fp.GetNumBits()):
        arr[i] = fp.GetBit(i)
    return arr


def get_descriptor_names_from_scaler(zscore_scaler) -> list[str]:
    if hasattr(zscore_scaler, "feature_names_in_"):
        return ["SMILES"] + list(zscore_scaler.feature_names_in_)
    return SELECTED_DESCRIPTOR_NAMES_FALLBACK


def generate_normalized_descriptors(
    df: pd.DataFrame,
    zscore_scaler,
    selected_descriptor_names: list[str],
) -> np.ndarray:
    all_descriptor_names = [name for name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptor_names)

    rows = []
    for smiles in df[SMILES_COLUMN]:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            rows.append(calculator.CalcDescriptors(mol))
        else:
            rows.append([np.nan] * len(all_descriptor_names))

    desc_df = pd.DataFrame(rows, columns=all_descriptor_names)
    desc_df.insert(0, "SMILES", df[SMILES_COLUMN].values)

    missing_desc = [c for c in selected_descriptor_names if c not in desc_df.columns]
    if missing_desc:
        raise ValueError(
            "Missing RDKit descriptors, possibly due to RDKit version differences: "
            f"{missing_desc}"
        )

    selected = desc_df[selected_descriptor_names]
    descriptor_numeric = selected.iloc[:, 1:]
    descriptor_numeric = descriptor_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)
    return zscore_scaler.transform(descriptor_numeric)


def build_all_features(
    df: pd.DataFrame,
    all_features: list[str],
    onehot_encoder,
    robust_scaler,
    minmax_scaler,
    zscore_scaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = clean_input_dataframe(df)

    print("Scaling numeric features...")
    X_robust = robust_scaler.transform(df[ROBUST_FEATURES])
    X_minmax = minmax_scaler.transform(df[MINMAX_FEATURES])
    X_numeric = np.concatenate([X_robust, X_minmax], axis=1)

    print("Encoding categorical features...")
    X_categorical = onehot_encoder.transform(df[CATEGORICAL_FEATURES])
    if hasattr(X_categorical, "toarray"):
        X_categorical = X_categorical.toarray()

    print("Generating Morgan fingerprints: radius=8, fpSize=2048...")
    X_morgan = np.array(df[SMILES_COLUMN].apply(smiles_to_morgan).tolist())

    print("Generating and scaling RDKit descriptors...")
    selected_descriptor_names = get_descriptor_names_from_scaler(zscore_scaler)
    X_descriptor = generate_normalized_descriptors(
        df=df,
        zscore_scaler=zscore_scaler,
        selected_descriptor_names=selected_descriptor_names,
    )

     X_all_array = np.concatenate(
        [X_numeric, X_categorical, X_morgan, X_descriptor],
        axis=1,
    )

    DROP_ZERO_COL_IDX = 15 

    if X_all_array.shape[1] == len(all_features) + 1:
        X_all_array = np.delete(X_all_array, DROP_ZERO_COL_IDX, axis=1)

    if X_all_array.shape[1] != len(all_features):
        raise ValueError(
            f"Feature dimension mismatch: generated {X_all_array.shape[1]} columns, "
            f"but all_features.txt contains {len(all_features)} names."
        )

    X_all = pd.DataFrame(X_all_array, columns=all_features, index=df.index)
    return X_all, df


def select_features_by_name(
    X_all: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    missing = [c for c in selected_features if c not in X_all.columns]
    if missing:
        raise ValueError(f"Selected features missing from X_all: {missing[:20]}")
    return X_all.loc[:, selected_features]


def save_results(results: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            results.where(pd.notna(results), None).to_dict(orient="records"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    model_dir = args.model_dir or (project_root / "models")
    output_dir = args.output_dir or (project_root / "outputs")
    input_path = args.input or (project_root / "data" / "example_input.json")

    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    input_path = input_path.resolve()

    output_prefix = args.output_prefix or input_path.stem
    output_csv = output_dir / f"{output_prefix}_predictions.csv"
    output_json = output_dir / f"{output_prefix}_predictions.json"

    print("Project root:", project_root)
    print("Model dir:", model_dir)
    print("Input:", input_path)
    print("Output CSV:", output_csv)

    all_features = read_feature_list(model_dir / "all_features.txt")
    mobility_selected_features = read_feature_list(
        model_dir / "mobility_selected_features.txt"
    )
    carrier_selected_features = read_feature_list(
        model_dir / "carrier_type_selected_features.txt"
    )

    bundle = joblib.load(require_file(model_dir / "preprocessing_bundle.pkl"))
    objects = bundle.get("objects", bundle) if isinstance(bundle, dict) else {}

    onehot_encoder = objects["onehot_encoder"]
    robust_scaler = objects["robust_scaler"]
    minmax_scaler = objects["minmax_scaler"]
    zscore_scaler = objects["zscore_scaler"]

    mobility_model = joblib.load(require_file(model_dir / "mobility_classifier.pkl"))
    carrier_model = joblib.load(require_file(model_dir / "carrier_type_classifier.pkl"))

    raw_df, active_json_path = load_prediction_input(input_path, output_dir)
    print("Active JSON input:", active_json_path)
    print("Input rows:", len(raw_df))

    X_all, input_df = build_all_features(
        df=raw_df,
        all_features=all_features,
        onehot_encoder=onehot_encoder,
        robust_scaler=robust_scaler,
        minmax_scaler=minmax_scaler,
        zscore_scaler=zscore_scaler,
    )

    X_mobility = select_features_by_name(X_all, mobility_selected_features)
    X_carrier = select_features_by_name(X_all, carrier_selected_features)

    mobility_pred = mobility_model.predict(X_mobility)
    carrier_pred = carrier_model.predict(X_carrier)

    mobility_proba = (
        mobility_model.predict_proba(X_mobility)
        if hasattr(mobility_model, "predict_proba")
        else None
    )
    carrier_proba = (
        carrier_model.predict_proba(X_carrier)
        if hasattr(carrier_model, "predict_proba")
        else None
    )

    results = pd.DataFrame(
        {
            "entry_id": (
                input_df["entry_id"]
                if "entry_id" in input_df.columns
                else [f"ENTRY_{i+1:03d}" for i in range(len(input_df))]
            ),
            "polymer_name": (
                input_df["polymer_name"] if "polymer_name" in input_df.columns else ""
            ),
            "pred_mobility_label": mobility_pred,
            "pred_carrier_type": carrier_pred,
        }
    )

    if mobility_proba is not None:
        for idx, cls in enumerate(mobility_model.classes_):
            results[f"mobility_probability_{cls}"] = mobility_proba[:, idx]

    if carrier_proba is not None:
        for idx, cls in enumerate(carrier_model.classes_):
            safe_cls = str(cls).replace(" ", "_").replace("/", "_")
            results[f"carrier_probability_{safe_cls}"] = carrier_proba[:, idx]

    save_results(results, output_csv, output_json)

    print("Saved CSV:", output_csv)
    print("Saved JSON:", output_json)
    print("Done.")


if __name__ == "__main__":
    main()
