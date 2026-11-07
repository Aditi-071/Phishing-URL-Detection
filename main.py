import argparse, os, re, json, warnings
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, classification_report)

warnings.filterwarnings("ignore")

def normalize_label(value):
    s = str(value).strip().lower()
    mapping = {
        "good": 0, "legitimate": 0, "benign": 0, "safe": 0, "normal": 0,
        "bad": 1, "phishing": 1, "phish": 1, "malicious": 1, "unsafe": 1
    }
    if s in mapping:
        return mapping[s]
    if s in ("0", "0.0"):
        return 0
    if s in ("1", "1.0"):
        return 1
    return np.nan

def safe_split(url):
    raw = str(url).strip()
    candidate = raw if "://" in raw else "http://" + raw
    try:
        return urlsplit(candidate)
    except Exception:
        return urlsplit("http://invalid.local")

def extract_features(url):
    raw = str(url).strip()
    p = safe_split(raw)
    host = p.hostname or ""
    path = p.path or ""
    query = p.query or ""
    lower = raw.lower()
    host_lower = host.lower()
    suspicious = ["login", "verify", "update", "secure", "account", "bank",
                  "confirm", "password", "signin", " paypal ", " wallet "]
    try:
        ip_host = int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host)))
    except Exception:
        ip_host = 0
    try:
        port = int(p.port is not None)
    except Exception:
        port = 0
    labels = [x for x in host.split(".") if x]
    features = {
        "url_length": len(raw),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "count_dots": raw.count("."),
        "count_hyphens": raw.count("-"),
        "count_underscores": raw.count("_"),
        "count_slashes": raw.count("/"),
        "count_question": raw.count("?"),
        "count_equal": raw.count("="),
        "count_at": raw.count("@"),
        "count_ampersand": raw.count("&"),
        "count_percent": raw.count("%"),
        "count_digits": sum(c.isdigit() for c in raw),
        "count_letters": sum(c.isalpha() for c in raw),
        "digit_ratio": sum(c.isdigit() for c in raw) / max(len(raw), 1),
        "letter_ratio": sum(c.isalpha() for c in raw) / max(len(raw), 1),
        "subdomain_count": max(len(labels) - 2, 0),
        "https": int(p.scheme.lower() == "https"),
        "ip_host": ip_host,
        "has_port": port,
        "double_slash": int("//" in path),
        "has_at_symbol": int("@" in raw),
        "encoded_percent": int("%" in raw),
        "suspicious_word_count": sum(w.strip() in lower for w in suspicious),
        "hyphen_in_hostname": int("-" in host_lower),
        "path_segments": len([x for x in path.split("/") if x]),
    }
    return features

def main():
    ap = argparse.ArgumentParser(description="Single-dataset phishing URL detection")
    ap.add_argument("--csv", default="data/phishing_site_urls_control_chars_removed.csv")
    ap.add_argument("--url-col", default="URL")
    ap.add_argument("--label-col", default="Label")
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}. Put your CSV in the data folder or pass --csv.")

    df = pd.read_csv(args.csv, low_memory=False)
    for col in (args.url_col, args.label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. CSV columns are: {list(df.columns)}")

    df = df[[args.url_col, args.label_col]].copy()
    df.columns = ["URL", "raw_label"]
    df["URL"] = df["URL"].astype(str).str.strip()
    df["target"] = df["raw_label"].apply(normalize_label)
    before = len(df)
    df = df.dropna(subset=["URL", "target"])
    df = df[df["URL"].str.len() > 0].copy()
    df["target"] = df["target"].astype(int)
    df = df.drop_duplicates(subset=["URL", "target"])
    # Remove URLs that appear with contradictory labels.
    conflicts = df.groupby("URL")["target"].nunique()
    conflict_urls = conflicts[conflicts > 1].index
    df = df[~df["URL"].isin(conflict_urls)].drop_duplicates("URL")
    if df["target"].nunique() != 2:
        raise ValueError("Both classes must be present. Check your label mapping.")
    audit = {
        "input_rows": int(before),
        "usable_rows": int(len(df)),
        "removed_or_invalid_rows": int(before - len(df)),
        "class_counts": {str(k): int(v) for k, v in df["target"].value_counts().sort_index().items()},
        "label_mapping": {"good/legitimate": 0, "bad/phishing": 1},
        "url_column": args.url_col,
        "label_column": args.label_col
    }
    with open("reports/data_quality_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    X = pd.DataFrame([extract_features(u) for u in df["URL"]])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=args.seed))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=args.seed, n_jobs=-1, min_samples_leaf=2
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=args.seed, n_jobs=-1, min_samples_leaf=2
        )
    }

    rows = []
    plt.figure()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        scores = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_phishing": precision_score(y_test, pred, zero_division=0),
            "recall_phishing": recall_score(y_test, pred, zero_division=0),
            "f1_phishing": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba)
        }
        rows.append(scores)
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Legitimate", "Phishing"])
        disp.plot(values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        safe = name.lower().replace(" ", "_")
        plt.savefig(f"reports/{safe}_confusion_matrix.png", bbox_inches="tight")
        plt.close()

        RocCurveDisplay.from_predictions(y_test, proba, name=name)
        plt.title("ROC Curves - Single Dataset")
        plt.savefig("reports/roc_curves.png", bbox_inches="tight")
        plt.close() if name == "Extra Trees" else None

        joblib.dump(model, f"models/{safe}.joblib")
        with open(f"reports/{safe}_classification_report.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, pred, target_names=["Legitimate", "Phishing"], zero_division=0))

        fitted = model.named_steps["model"] if isinstance(model, Pipeline) else model
        if hasattr(fitted, "feature_importances_"):
            imp = pd.DataFrame({"feature": X.columns, "importance": fitted.feature_importances_})
            imp.sort_values("importance", ascending=False).to_csv(
                f"reports/{safe}_feature_importance.csv", index=False
            )

    results = pd.DataFrame(rows).sort_values("f1_phishing", ascending=False)
    results.to_csv("reports/model_comparison.csv", index=False)
    print("\n=== MODEL COMPARISON ===")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nCompleted. Check the reports/ and models/ folders.")
    print("Reminder: this is a random holdout from one dataset, not cross-dataset or future-time validation.")

if __name__ == "__main__":
    main()
