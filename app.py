import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

st.set_page_config(page_title="Major Recommendation (Cambodia)", layout="wide")
st.title("🎓 Major Recommendation — Cambodia Scenario")
st.caption("Predict and rank majors using model similarity + market metrics (Popularity, Success, Salary, Demand)")

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pickle(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(p)

@st.cache_resource(show_spinner=False)
def load_csv(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(p)

try:
    loaded_model = load_pickle("model/rf_model.pkl")
    # This may be a dict of encoders or anything you saved
    try:
        loaded_encoders = load_pickle("model/feature_encoders.pkl")
    except Exception:
        loaded_encoders = None  # ok if your model is a Pipeline
    loaded_target_le = load_pickle("model/target_encoder.pkl")

    df = load_csv("cambodia_major_dataset_v18_50majors.csv")
    major_stats = load_csv("major_stats_cambodia_v1.csv")
except Exception as e:
    st.error(f"Startup load error: {e}")
    st.stop()

# -----------------------------
# Helper: compute final scores
# -----------------------------
def compute_final_scores(
    similarity_df: pd.DataFrame,
    major_stats: pd.DataFrame,
    w_sim=0.5, w_pop=0.1, w_succ=0.2, w_sal=0.1, w_dem=0.1,
    right_key: str | None = None
) -> pd.DataFrame:
    # Auto-detect the right join key if not supplied
    if right_key is None:
        if "Recommend_Major" in major_stats.columns:
            right_key = "Recommend_Major"
        elif "Major" in major_stats.columns:
            right_key = "Major"
        else:
            st.error("major_stats must contain either 'Recommend_Major' or 'Major' column.")
            return similarity_df.assign(Final_Score=np.nan)

    merged = similarity_df.merge(major_stats, left_on="Major", right_on=right_key, how="left")

    # Ensure score columns exist
    needed = ["Popularity", "SuccessRate", "Salary", "Demand"]
    missing_cols = [c for c in needed if c not in merged.columns]
    if missing_cols:
        st.error(f"Missing columns in major_stats for scoring: {missing_cols}")
        # Return with NaN Final_Score so UI still renders
        return merged.assign(Final_Score=np.nan)

    # Warn if any majors missing and fill numeric NaNs by median
    if merged[needed].isna().any().any():
        missing = merged.loc[merged["Popularity"].isna(), "Major"].dropna().unique().tolist()
        if missing:
            st.warning(f"Majors missing in major_stats (filled by medians): {missing}")
        for c in needed:
            if merged[c].isna().any():
                merged[c] = merged[c].fillna(merged[c].median())

    if "Category" not in merged.columns:
        merged["Category"] = "Unknown"

    merged["Final_Score"] = (
        w_sim * merged["Similarity"]
        + w_pop * merged["Popularity"]
        + w_succ * merged["SuccessRate"]
        + w_sal * merged["Salary"]
        + w_dem * merged["Demand"]
    )

    cols = ["Major", "Final_Score", "Similarity", "Popularity", "SuccessRate", "Salary", "Demand", "Category"]
    # Keep only columns that actually exist to avoid KeyError if Category/others are absent
    cols = [c for c in cols if c in merged.columns]
    return merged.sort_values("Final_Score", ascending=False)[cols].reset_index(drop=True)

# -----------------------------
# Sidebar: Weights
# -----------------------------
st.sidebar.header("⚙️ Configuration Weights")
w_sim = st.sidebar.slider("Similarity", 0.0, 1.0, 0.5, 0.05)
w_pop = st.sidebar.slider("Popularity", 0.0, 1.0, 0.1, 0.05)
w_succ = st.sidebar.slider("SuccessRate", 0.0, 1.0, 0.2, 0.05)
w_sal = st.sidebar.slider("Salary", 0.0, 1.0, 0.1, 0.05)
w_dem = st.sidebar.slider("Demand", 0.0, 1.0, 0.1, 0.05)

# -----------------------------
# New Student Form
# -----------------------------
st.subheader("🧑‍🎓 Personal Info")
col1, col2, col3, col4 = st.columns(4)

# Collect inputs
with col1:
    gender_val = st.selectbox("Gender", options=sorted(df["Gender"].dropna().unique().tolist()), index=0)
with col2:
    province_val = st.selectbox("Province", options=sorted(df["Province"].dropna().unique().tolist()), index=0)
with col3:
    school_val = st.selectbox("School_Type", options=sorted(df["School_Type"].dropna().unique().tolist()), index=0)
with col4:
    parent_val = st.selectbox("Parent_Expectation", options=sorted(df["Parent_Expectation"].dropna().unique().tolist()), index=0)

st.subheader("📚 Academic Background")
stream_val = st.selectbox("Stream", options=sorted(df["Stream"].dropna().unique().tolist()), index=0)
col1, col2, col3, col4 = st.columns(4)
with col1: 
    kh_val = st.selectbox("Khmer Grade", options=["A", "B", "C", "D", "E", "F"])
with col2:
    en_val = st.selectbox("English Grade", options=["A", "B", "C", "D", "E", "F"])
with col3:
    math_val = st.selectbox("Math Grade", options=["A", "B", "C", "D", "E", "F"])
with col4:
    hist_val = st.selectbox("History Grade", options=["A", "B", "C", "D", "E", "F"])

# Stream-specific subjects
if stream_val == "Science":
    with col1:
        phy_val = st.selectbox("Physics Grade", options=["A", "B", "C", "D", "E", "F"])
    with col2:
        ch_val = st.selectbox("Chemistry Grade", options=["A", "B", "C", "D", "E", "F"])
    with col3:
        bio_val = st.selectbox("Biology Grade", options=["A", "B", "C", "D", "E", "F"])
else:
    with col1:
        geo_val = st.selectbox("Geography Grade", options=["A", "B", "C", "D", "E", "F"])
    with col2:
        mo_val = st.selectbox("Morality Grade", options=["A", "B", "C", "D", "E", "F"])
    with col3:
        ea_val = st.selectbox("Earth Science Grade", options=["A", "B", "C", "D", "E", "F"])

st.subheader("📊 Interest & Personality")
col5, col6, col7, col8, col9 = st.columns(5)
with col5:
    int_tech_val = st.selectbox("Interest in Technology", options=sorted(df["Interest_Tech"].dropna().unique().tolist()), index=0)
    careerg_val = st.selectbox("Career_Goal", options=sorted(df["Career_Goal"].dropna().unique().tolist()), index=0)
with col6:
    int_bus_val = st.selectbox("Interest in Business", options=sorted(df["Interest_Business"].dropna().unique().tolist()), index=0)
    workpref_val = st.selectbox("WorkPreference", options=sorted(df["WorkPreference"].dropna().unique().tolist()), index=0)
with col7:
    int_hea_val = st.selectbox("Interest in Health", options=sorted(df["Interest_Health"].dropna().unique().tolist()), index=0)
    personality_val = st.selectbox("Personality", options=sorted(df["Personality"].dropna().unique().tolist()), index=0)
with col8:
    int_desi_val = st.selectbox("Interest in Design", options=sorted(df["Interest_Design"].dropna().unique().tolist()), index=0)
with col9:
    int_law_val = st.selectbox("Interest in Law", options=sorted(df["Interest_Law"].dropna().unique().tolist()), index=0)

# -----------------------------
# Safe prediction helpers
# -----------------------------
def is_pipeline(model) -> bool:
    # Basic check for sklearn Pipeline
    return hasattr(model, "steps") and any(hasattr(step, "fit") for _, step in model.steps)

def encode_with_loaded_encoders(df_in: pd.DataFrame, encoders) -> pd.DataFrame:
    """
    Apply saved encoders if provided (assumes dict-like {col: fitted_LabelEncoder or mapping}).
    Unseen labels -> filled as 0 (or you can choose another policy).
    """
    if encoders is None:
        return df_in

    X = df_in.copy()
    # Accept dict-like objects with .items()
    items = encoders.items() if hasattr(encoders, "items") else []
    for col, enc in items:
        if col not in X.columns:
            continue
        series = X[col].astype(object)

        # LabelEncoder-like
        if hasattr(enc, "classes_"):
            mapping = {cls: i for i, cls in enumerate(enc.classes_)}
            X[col] = series.map(mapping)
            # Handle unseen / NaN
            X[col] = X[col].fillna(0).astype(int)
        # Mapping-like
        elif isinstance(enc, dict):
            X[col] = series.map(enc).fillna(0).astype(int)
        else:
            # If unknown type, leave as-is (may error if the model expects numeric)
            pass

    return X

# -----------------------------
# Action: Recommend
# -----------------------------
if st.button("🔮 Recommend Majors", type="primary"):
    # Build new student row aligned to training features
    new_row = {
        "Gender": gender_val,
        "Province": province_val,
        "School_Type": school_val,
        "Parent_Expectation": parent_val,
        "Stream": stream_val,
        "Khmer": kh_val,
        "English": en_val,
        "Math": math_val,
        "Physics": locals().get("phy_val", np.nan),
        "Chemistry": locals().get("ch_val", np.nan),
        "Biology": locals().get("bio_val", np.nan),
        "History": hist_val,
        "Geography": locals().get("geo_val", np.nan),
        "Morality": locals().get("mo_val", np.nan),
        "Earth": locals().get("ea_val", np.nan),
        "Interest_Tech": int_tech_val,
        "Interest_Business": int_bus_val,
        "Interest_Health": int_hea_val,
        "Interest_Design": int_desi_val,
        "Interest_Law": int_law_val,
        "Personality": personality_val,
        "WorkPreference": workpref_val,
        "Career_Goal": careerg_val,
    }
    new_student = pd.DataFrame([new_row])

    try:
        # If your model is a Pipeline, it should handle encoding internally.
        if is_pipeline(loaded_model):
            X_for_pred = new_student
        else:
            # Fall back to manual encoding using loaded_encoders if provided
            X_for_pred = encode_with_loaded_encoders(new_student, loaded_encoders)

        # Try to align to model's expected feature order if available
        if hasattr(loaded_model, "feature_names_in_"):
            expected = list(loaded_model.feature_names_in_)
            missing_expected = [c for c in expected if c not in X_for_pred.columns]
            if missing_expected:
                st.warning(f"Adding missing expected features with NaN: {missing_expected}")
                for c in missing_expected:
                    X_for_pred[c] = np.nan
            # Extra columns will be ignored only if the estimator allows it; safest to select expected
            X_for_pred = X_for_pred[expected]

        proba = loaded_model.predict_proba(X_for_pred)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    # Convert probabilities to table
    major_names = getattr(loaded_target_le, "classes_", None)
    if major_names is None:
        st.error("target_encoder does not expose classes_.")
        st.stop()

    top_df = pd.DataFrame({"Major": major_names, "Similarity_raw": proba})
    # Min-max scale probabilities to [0,1]; if all equal, set scaled to 0
    raw = top_df["Similarity_raw"].astype(float)
    if raw.max() - raw.min() > 0:
        top_df["Similarity"] = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        top_df["Similarity"] = 0.0
    top_df = top_df.sort_values("Similarity", ascending=False).reset_index(drop=True)

    st.subheader("Top 10 by Model Similarity")
    st.dataframe(top_df.head(10), use_container_width=True)

    # Compute final score using major_stats and selected weights
    scored = compute_final_scores(top_df, major_stats, w_sim=w_sim, w_pop=w_pop, w_succ=w_succ, w_sal=w_sal, w_dem=w_dem)
    st.subheader("🏆 Final Recommendations (with market metrics)")
    st.dataframe(scored.head(5), use_container_width=True)

    # Download
    csv = scored.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download full ranking (CSV)",
        data=csv,
        file_name="major_recommendations_ranked.csv",
        mime="text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("© 2025 Major Recommender • RandomForest, Cambodia-aligned stats • Built with Streamlit")
