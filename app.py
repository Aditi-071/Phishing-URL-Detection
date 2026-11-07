import re
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import pandas as pd
import streamlit as st


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PhishGuard - URL Detection",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Random Forest": "random_forest.joblib",
    "Extra Trees": "extra_trees.joblib"
}


# =====================================================
# 2. URL PARSING
# Exact logic from main.py
# =====================================================

def safe_split(url):
    raw = str(url).strip()
    candidate = raw if "://" in raw else "http://" + raw

    try:
        return urlsplit(candidate)
    except Exception:
        return urlsplit("http://invalid.local")


# =====================================================
# 3. FEATURE EXTRACTION
# Exact logic from main.py
# =====================================================

def extract_features(url):

    raw = str(url).strip()
    p = safe_split(raw)

    host = p.hostname or ""
    path = p.path or ""
    query = p.query or ""

    lower = raw.lower()
    host_lower = host.lower()

    suspicious = [
        "login", "verify", "update", "secure", "account",
        "bank", "confirm", "password", "signin",
        " paypal ", " wallet "
    ]

    try:
        ip_host = int(
            bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))
        )
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
        "suspicious_word_count": sum(
            w.strip() in lower for w in suspicious
        ),
        "hyphen_in_hostname": int("-" in host_lower),
        "path_segments": len(
            [x for x in path.split("/") if x]
        )
    }

    return features


# =====================================================
# 4. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource
def load_models():

    models = {}

    for model_name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            st.warning(f"Model not found: {filename}")
            continue

        try:
            models[model_name] = joblib.load(model_path)

        except Exception as error:
            st.error(f"Could not load {model_name}: {error}")

    return models


# =====================================================
# 5. PREDICTION FUNCTION
# =====================================================

def predict_url(model, features):

    input_data = pd.DataFrame([features])

    prediction = int(model.predict(input_data)[0])

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            probability = float(
                probabilities[classes.index(1)]
            )

    return prediction, probability


# =====================================================
# 6. HEADER
# =====================================================

st.title("🛡️ PhishGuard")

st.subheader("Machine Learning Based Phishing URL Detection")

st.write(
    "Analyse website URLs using three trained machine learning "
    "algorithms: Logistic Regression, Random Forest and Extra Trees."
)

st.info(
    "Enter a URL to analyse its characteristics. "
    "This application does not visit or open the entered website."
)

st.markdown("---")


# =====================================================
# 7. URL INPUT
# =====================================================

url = st.text_input(
    "🌐 Enter Website URL",
    placeholder="https://www.google.com"
)

check_button = st.button(
    "🔍 Analyse URL",
    type="primary",
    use_container_width=True
)


# =====================================================
# 8. ANALYSIS
# =====================================================

if check_button:

    if not url.strip():

        st.warning("Please enter a URL.")

    else:

        try:

            parsed = safe_split(url)

            if not parsed.hostname:
                st.error("Please enter a valid URL.")
                st.stop()

            # Extract exactly the same features as training
            features = extract_features(url)

            # Load existing trained models
            models = load_models()

            if not models:
                st.error(
                    "No trained models could be loaded. "
                    "Check your models folder."
                )
                st.stop()

            # =========================================
            # 9. PREDICTION RESULTS
            # =========================================

            st.markdown("---")
            st.header("📊 Prediction Results")

            st.write(f"**Entered URL:** {url}")

            results = []

            for model_name, model in models.items():

                try:

                    prediction, probability = predict_url(
                        model, features
                    )

                    label = (
                        "Phishing"
                        if prediction == 1
                        else "Legitimate"
                    )

                    results.append({
                        "Model": model_name,
                        "Prediction": label,
                        "Probability": probability
                    })

                except Exception as error:

                    st.error(
                        f"{model_name} prediction failed: {error}"
                    )

            # =========================================
            # 10. INDIVIDUAL MODEL RESULTS
            # =========================================

            if results:

                columns = st.columns(len(results))

                for column, result in zip(columns, results):

                    with column:

                        st.subheader(result["Model"])

                        if result["Prediction"] == "Phishing":

                            st.error("🚨 Phishing URL Detected!")

                        else:

                            st.success("✅ Predicted as Legitimate")

                        probability = result["Probability"]

                        if probability is not None:

                            st.metric(
                                "Phishing Probability",
                                f"{probability * 100:.2f}%"
                            )

                            st.progress(
                                probability,
                                text="Phishing score"
                            )

                        else:

                            st.caption(
                                "Probability not available"
                            )

            # =========================================
            # 11. MODEL COMPARISON
            # =========================================

            if results:

                st.markdown("---")
                st.header("📋 Model Comparison")

                comparison_data = []

                for result in results:

                    probability = result["Probability"]

                    comparison_data.append({
                        "Model": result["Model"],
                        "Prediction": result["Prediction"],
                        "Phishing Probability": (
                            f"{probability * 100:.2f}%"
                            if probability is not None
                            else "Unavailable"
                        )
                    })

                comparison = pd.DataFrame(comparison_data)

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

            # =========================================
            # 12. MODEL AGREEMENT
            # =========================================

            if results:

                st.markdown("---")
                st.header("🗳️ Overall Model Agreement")

                phishing_votes = sum(
                    r["Prediction"] == "Phishing"
                    for r in results
                )

                legitimate_votes = sum(
                    r["Prediction"] == "Legitimate"
                    for r in results
                )

                total_votes = len(results)

                st.write(
                    f"**Phishing predictions:** "
                    f"{phishing_votes}/{total_votes}"
                )

                st.write(
                    f"**Legitimate predictions:** "
                    f"{legitimate_votes}/{total_votes}"
                )

                if phishing_votes > total_votes / 2:

                    st.error(
                        "Majority of models predict this URL as phishing."
                    )

                elif legitimate_votes > total_votes / 2:

                    st.success(
                        "Majority of models predict this URL as legitimate."
                    )

                else:

                    st.warning(
                        "The models are equally divided."
                    )

                if phishing_votes > 0 and legitimate_votes > 0:

                    st.warning(
                        "The models disagree. Treat this prediction "
                        "as uncertain rather than relying on majority voting."
                    )

            # =========================================
            # 13. EXTRACTED URL FEATURES
            # ALWAYS VISIBLE
            # =========================================

            st.markdown("---")
            st.header("🔎 Extracted URL Features")

            st.write(
                "The following 27 features were extracted from "
                "your entered URL and supplied to the trained models."
            )

            feature_table = pd.DataFrame(
                list(features.items()),
                columns=["Feature Name", "Extracted Value"]
            )

            st.dataframe(
                feature_table,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Successfully extracted {len(features)} URL features."
            )

        except Exception as error:

            st.error(f"An error occurred: {error}")


# =====================================================
# 14. ABOUT SECTION
# =====================================================

st.markdown("---")

st.header("About PhishGuard")

st.write(
    "PhishGuard is a machine learning project that compares "
    "three classification algorithms for phishing URL detection."
)

st.write(
    "**Algorithms:** Logistic Regression, Random Forest and Extra Trees."
)

st.caption(
    "Disclaimer: Predictions are based on URL characteristics. "
    "A legitimate prediction does not guarantee that a website is safe. "
    "Model probabilities are not guaranteed real-world risk estimates."
)

st.caption(
    "PhishGuard | Machine Learning Based Phishing URL Detection"
)