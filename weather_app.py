import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(
    page_title="WeatherAUS – EDA Dashboard",
    layout="wide"
)

st.title("🌦️ WeatherAUS – EDA & Preprocessing Dashboard")
st.write("This app reproduces the main EDA & preprocessing steps from the notebook.")


@st.cache_data
def load_raw_data():
    df = pd.read_csv("weatherAUS.csv")
    return df


@st.cache_data
def prepare_data():
    df = load_raw_data().copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype("category")

    missing_pct = df.isna().mean() * 100

    cols_to_drop_rows = missing_pct[missing_pct < 5].index.tolist()
    if cols_to_drop_rows:
        df = df.dropna(subset=cols_to_drop_rows)

    high_missing_cols = missing_pct[missing_pct > 40].index.tolist()
    for col in high_missing_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    num_fill_cols = ["WindGustSpeed", "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm"]
    for col in num_fill_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    cat_fill_cols = ["WindGustDir", "WindDir9am"]
    for col in cat_fill_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    df = df.dropna(axis=1, how="all")

    if "Date" in df.columns:
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfYear"] = df["Date"].dt.dayofyear

        def month_to_season(m):
            if m in [12, 1, 2]:
                return "Summer"
            elif m in [3, 4, 5]:
                return "Autumn"
            elif m in [6, 7, 8]:
                return "Winter"
            else:
                return "Spring"

        df["Season"] = df["Month"].apply(month_to_season)

    if "RainTomorrow" in df.columns:
        rt = df["RainTomorrow"].astype(str).str.strip().str.upper()
        df["RainTomorrow_bin"] = rt.map({"YES": "Yes", "NO": "No"})
    if "RainToday" in df.columns:
        rtd = df["RainToday"].astype(str).str.strip().str.upper()
        df["RainToday_bin"] = rtd.map({"YES": "Yes", "NO": "No"})

    if {"MaxTemp", "MinTemp"}.issubset(df.columns):
        df["TempDiff"] = df["MaxTemp"] - df["MinTemp"]
    if {"Pressure9am", "Pressure3pm"}.issubset(df.columns):
        df["PressureDrop"] = df["Pressure9am"] - df["Pressure3pm"]
    if {"Humidity3pm", "Humidity9am"}.issubset(df.columns):
        df["HumidityDiff"] = df["Humidity3pm"] - df["Humidity9am"]
    if {"WindSpeed3pm", "WindSpeed9am"}.issubset(df.columns):
        df["WindSpeedChange"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
    if "WindGustSpeed" in df.columns:
        df["IsWindyDay"] = (df["WindGustSpeed"] > 60).astype(int)

    return df



df_raw = load_raw_data()
df = prepare_data()

with st.sidebar:
    st.header("Dataset Info")
    st.write(f"**Raw shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.write(f"**After cleaning:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("File used: `weatherAUS.csv`")
    st.markdown("---")

    st.header("Filters")

    
    if "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
        selected_locations = st.multiselect(
            "Location(s):",
            options=locations,
            default=locations[:5] if len(locations) > 5 else locations
        )
    else:
        selected_locations = None

    
    if "Season" in df.columns:
        seasons = df["Season"].dropna().unique().tolist()
        selected_seasons = st.multiselect(
            "Season(s):",
            options=seasons,
            default=seasons
        )
    else:
        selected_seasons = None

    
    if "RainTomorrow_bin" in df.columns:
        target_opts = ["Yes", "No"]
        selected_target = st.multiselect(
            "Rain Tomorrow?",
            options=target_opts,
            default=target_opts
        )
    else:
        selected_target = None

df_filtered = df.copy()

if selected_locations:
    df_filtered = df_filtered[df_filtered["Location"].isin(selected_locations)]

if selected_seasons:
    df_filtered = df_filtered[df_filtered["Season"].isin(selected_seasons)]

if selected_target:
    df_filtered = df_filtered[df_filtered["RainTomorrow_bin"].isin(selected_target)]



tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Overview",
    "🕳 Missing Values",
    "📈 Distributions",
    "🔥 Correlations",
    "📍 Target & Locations"
])


with tab1:
    st.subheader("Data Preview (Filtered)")
    st.dataframe(df_filtered.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Numeric Summary (Filtered)")
        num_desc = df_filtered.select_dtypes(include=["number"]).describe().T
        st.dataframe(num_desc)
    with col2:
        st.subheader("Categorical Summary (Filtered)")
        cat_cols = df_filtered.select_dtypes(include=["category", "object"]).columns
        if len(cat_cols) > 0:
            cat_desc = df_filtered[cat_cols].describe().T
            st.dataframe(cat_desc)
        else:
            st.info("No categorical columns found.")


with tab2:
    st.subheader("Missing Values (Before Cleaning – Raw Data)")
    missing_raw = df_raw.isna().sum().reset_index()
    missing_raw.columns = ["Column", "MissingCount"]
    missing_raw["MissingPct"] = missing_raw["MissingCount"] / len(df_raw) * 100

    st.dataframe(missing_raw.sort_values("MissingPct", ascending=False))

    fig_mv = px.bar(
        missing_raw.sort_values("MissingPct", ascending=False),
        x="Column",
        y="MissingPct",
        title="Missing Values Percentage (Raw Data)"
    )
    fig_mv.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_mv, use_container_width=True)


with tab3:
    st.subheader("Numeric Feature Distribution (Filtered)")

    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        selected_num = st.selectbox("Choose a numeric column:", numeric_cols)
        col_left, col_right = st.columns(2)

        with col_left:
            st.write("Histogram + Boxplot (Plotly)")
            fig_hist = px.histogram(
                df_filtered,
                x=selected_num,
                nbins=40,
                marginal="box",
                opacity=0.8,
                title=f"Distribution of {selected_num}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_right:
            st.write("Seaborn Boxplot (IQR / Outlier View)")
            fig, ax = plt.subplots()
            sns.boxplot(x=df_filtered[selected_num], ax=ax)
            ax.set_title(f"Boxplot of {selected_num}")
            st.pyplot(fig)
    else:
        st.info("No numeric columns found.")


with tab4:
    st.subheader("Correlation Heatmap (Numeric, Filtered)")

    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) > 1:
        corr = df_filtered[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)")
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")


with tab5:
    st.subheader("Target Distribution – RainTomorrow_bin (Filtered)")

    if "RainTomorrow_bin" in df_filtered.columns:
        target_counts = df_filtered["RainTomorrow_bin"].value_counts().reset_index()
        target_counts.columns = ["RainTomorrow_bin", "Count"]
        target_counts["Pct"] = target_counts["Count"] / target_counts["Count"].sum() * 100

        st.dataframe(target_counts)

        fig_tgt = px.bar(
            target_counts,
            x="RainTomorrow_bin",
            y="Count",
            text="Pct",
            title="Class Balance: RainTomorrow_bin"
        )
        fig_tgt.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(fig_tgt, use_container_width=True)
    else:
        st.info("Column `RainTomorrow_bin` not found – check preprocessing.")

    st.subheader("Top Locations by Rain Tomorrow Rate (Filtered)")
    if {"Location", "RainTomorrow_bin"}.issubset(df_filtered.columns):
        loc_rain = (
            df_filtered.assign(
                RainTomorrowFlag=df_filtered["RainTomorrow_bin"].eq("Yes").astype(int)
            )
            .groupby("Location")["RainTomorrowFlag"]
            .mean()
            .reset_index()
            .rename(columns={"RainTomorrowFlag": "RainTomorrowRate"})
            .sort_values("RainTomorrowRate", ascending=False)
            .head(15)
        )

        st.dataframe(loc_rain)

        fig_loc = px.bar(
            loc_rain,
            x="Location",
            y="RainTomorrowRate",
            title="Top 15 Locations by Probability of Rain Tomorrow"
        )
        fig_loc.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_loc, use_container_width=True)
    else:
        st.info("Need `Location` and `RainTomorrow_bin` columns for this analysis.")
