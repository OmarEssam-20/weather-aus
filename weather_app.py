import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="WeatherAUS – EDA Dashboard",
    layout="wide"
)

st.title("🌦️ WeatherAUS – EDA & Preprocessing Dashboard")
st.write("This app reproduces the main EDA & preprocessing steps from the notebook.")


# =========================
# Data Loading & Prep
# =========================
@st.cache_data
def load_raw_data():
    df = pd.read_csv("weatherAUS.csv")
    return df


@st.cache_data
def prepare_data():
    df = load_raw_data().copy()

    # ---- Convert Date & dtypes ----
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype("category")

    # ---- Drop rows with NaN in key columns (similar to EDA notebook) ----
    cols_to_drop_rows = [
        "MinTemp", "MaxTemp", "Rainfall",
        "WindDir3pm", "WindSpeed9am", "WindSpeed3pm",
        "Humidity3pm", "Humidity9am",
        "Temp9am", "Temp3pm",
        "RainToday", "RainTomorrow"
    ]
    cols_to_drop_rows = [c for c in cols_to_drop_rows if c in df.columns]
    if cols_to_drop_rows:
        df = df.dropna(subset=cols_to_drop_rows)

    # ---- Drop columns with >40% missing (Evaporation, Sunshine) ----
    cols_to_drop_cols = ["Evaporation", "Sunshine"]
    cols_to_drop_cols = [c for c in cols_to_drop_cols if c in df.columns]
    if cols_to_drop_cols:
        df.drop(columns=cols_to_drop_cols, inplace=True)

    # ---- Fill remaining numeric missing with median ----
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # ---- Fill selected categorical columns with mode ----
    cat_fill_cols = ["WindGustDir", "WindDir9am"]
    for col in cat_fill_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)

    # ---- Drop any all-NaN columns just in case ----
    df = df.dropna(axis=1, how="all")

    # ---- Date-based features ----
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

    # ---- Clean RainToday & RainTomorrow (normalize to 'Yes'/'No' strings) ----
    if "RainTomorrow" in df.columns:
        rt = df["RainTomorrow"].astype(str).str.strip().str.upper()
        df["RainTomorrow"] = rt.map({"YES": "Yes", "NO": "No"})
    if "RainToday" in df.columns:
        rtd = df["RainToday"].astype(str).str.strip().str.upper()
        df["RainToday"] = rtd.map({"YES": "Yes", "NO": "No"})

    # ---- Feature engineering ----
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


# =========================
# Load Data
# =========================
df_raw = load_raw_data()
df = prepare_data()

# =========================
# Sidebar Filters
# =========================
with st.sidebar:
    st.header("Dataset Info")
    st.write(f"**Raw shape:** {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
    st.write(f"**After cleaning:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("File used: `weatherAUS.csv`")
    st.markdown("---")

    st.header("Filters")

    # Location filter
    if "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
        selected_locations = st.multiselect(
            "Location(s):",
            options=locations,
            default=locations[:5] if len(locations) > 5 else locations
        )
    else:
        selected_locations = None

    # Season filter
    if "Season" in df.columns:
        seasons = df["Season"].dropna().unique().tolist()
        selected_seasons = st.multiselect(
            "Season(s):",
            options=seasons,
            default=seasons
        )
    else:
        selected_seasons = None

    # RainTomorrow filter (using RainTomorrow directly)
    if "RainTomorrow" in df.columns:
        target_opts = ["Yes", "No"]
        selected_target = st.multiselect(
            "Rain Tomorrow?",
            options=target_opts,
            default=target_opts
        )
    else:
        selected_target = None

# Apply filters
df_filtered = df.copy()

if selected_locations:
    df_filtered = df_filtered[df_filtered["Location"].isin(selected_locations)]

if selected_seasons:
    df_filtered = df_filtered[df_filtered["Season"].isin(selected_seasons)]

if selected_target:
    df_filtered = df_filtered[df_filtered["RainTomorrow"].isin(selected_target)]


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📄 Overview",
    "🕳 Missing Values",
    "📈 Distributions",
    "🔥 Correlations",
    "📍 Target & Locations",
    "🔍 Insights"
])

# -------------------------
# Tab 1: Overview
# -------------------------
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


# -------------------------
# Tab 2: Missing Values
# -------------------------
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


# -------------------------
# Tab 3: Distributions
# -------------------------
with tab3:
    st.subheader("Numeric Feature Distribution (Filtered)")

    numeric_cols = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        selected_num = st.selectbox("Choose a numeric column:", numeric_cols)
        col_left, col_right = st.columns(2)

        # Left: Histogram (Plotly)
        with col_left:
            st.write("Histogram (Plotly)")
            fig_hist = px.histogram(
                df_filtered,
                x=selected_num,
                nbins=40,
                opacity=0.8,
                title=f"Distribution of {selected_num}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Right: Density Plot (KDE) بدل الـ Boxplot
        with col_right:
            st.write("Density Plot (KDE)")
            fig, ax = plt.subplots()
            sns.kdeplot(
                df_filtered[selected_num].dropna(),
                ax=ax,
                fill=True
            )
            ax.set_title(f"Density Plot of {selected_num}")
            ax.set_xlabel(selected_num)
            st.pyplot(fig)
    else:
        st.info("No numeric columns found.")


# -------------------------
# Tab 4: Correlations
# -------------------------
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


# -------------------------
# Tab 5: Target & Locations
# -------------------------
with tab5:
    st.subheader("Target Distribution – RainTomorrow (Filtered)")

    if "RainTomorrow" in df_filtered.columns:
        target_counts = df_filtered["RainTomorrow"].value_counts().reset_index()
        target_counts.columns = ["RainTomorrow", "Count"]
        target_counts["Pct"] = target_counts["Count"] / target_counts["Count"].sum() * 100

        st.dataframe(target_counts)

        fig_tgt = px.bar(
            target_counts,
            x="RainTomorrow",
            y="Count",
            text="Pct",
            title="Class Balance: RainTomorrow"
        )
        fig_tgt.update_traces(texttemplate="%{text:.1f}%")
        st.plotly_chart(fig_tgt, use_container_width=True)
    else:
        st.info("Column `RainTomorrow` not found – check preprocessing.")

    st.subheader("Top 5 Locations by Rain Tomorrow Rate (Filtered)")
    if {"Location", "RainTomorrow"}.issubset(df_filtered.columns):
        loc_rain = (
            df_filtered.assign(
                RainTomorrowFlag=df_filtered["RainTomorrow"].eq("Yes").astype(int)
            )
            .groupby("Location")["RainTomorrowFlag"]
            .mean()
            .reset_index()
            .rename(columns={"RainTomorrowFlag": "RainTomorrowRate"})
            .sort_values("RainTomorrowRate", ascending=False)
            .head(5)
        )

        st.dataframe(loc_rain)

        fig_loc = px.bar(
            loc_rain,
            x="Location",
            y="RainTomorrowRate",
            title="Top 5 Locations by Probability of Rain Tomorrow"
        )
        fig_loc.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_loc, use_container_width=True)
    else:
        st.info("Need `Location` and `RainTomorrow` columns for this analysis.")


# -------------------------
# Tab 6: Insights
# -------------------------
with tab6:
    st.subheader("Key Rain Insights (Filtered Data)")

    if "RainTomorrow" not in df_filtered.columns:
        st.info("`RainTomorrow` not found – cannot compute insights.")
    else:
        # Create RainFlag (0/1) for convenience
        df_ins = df_filtered.copy()
        df_ins["RainFlag"] = df_ins["RainTomorrow"].eq("Yes").astype(int)

        insight = st.selectbox(
            "Choose an insight:",
            [
                "Windy vs Non-Windy Days",
                "Season vs Rain Probability",
                "Humidity at 3pm vs RainTomorrow",
                "Temperature Difference vs RainTomorrow",
                "Top 5 Locations by Rain Probability"
            ]
        )

        # 1) Windy vs Non-Windy
        if insight == "Windy vs Non-Windy Days":
            if "IsWindyDay" in df_ins.columns:
                windy_rain = (
                    df_ins.groupby("IsWindyDay")["RainFlag"]
                          .mean()
                          .reset_index()
                          .rename(columns={"RainFlag": "RainTomorrowRate"})
                )
                st.write("**Rain probability on windy vs non-windy days:**")
                st.dataframe(windy_rain)

                fig = px.bar(
                    windy_rain,
                    x="IsWindyDay",
                    y="RainTomorrowRate",
                    title="Rain Probability by Windy (1) vs Non-Windy (0) Days"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("`IsWindyDay` feature not available.")

        # 2) Season vs Rain Probability
        elif insight == "Season vs Rain Probability":
            if "Season" in df_ins.columns:
                season_rain = (
                    df_ins.groupby("Season")["RainFlag"]
                          .mean()
                          .reset_index()
                          .rename(columns={"RainFlag": "RainTomorrowRate"})
                          .sort_values("RainTomorrowRate", ascending=False)
                )
                st.write("**Rain probability by season:**")
                st.dataframe(season_rain)

                fig = px.bar(
                    season_rain,
                    x="Season",
                    y="RainTomorrowRate",
                    title="Rain Probability by Season"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("`Season` feature not available.")

        # 3) Humidity at 3pm vs RainTomorrow
        elif insight == "Humidity at 3pm vs RainTomorrow":
            if "Humidity3pm" in df_ins.columns:
                st.write("**Humidity at 3pm grouped by RainTomorrow:**")
                desc = df_ins.groupby("RainTomorrow")["Humidity3pm"].describe()
                st.dataframe(desc)

                fig, ax = plt.subplots()
                sns.boxplot(
                    data=df_ins,
                    x="RainTomorrow",
                    y="Humidity3pm",
                    ax=ax
                )
                ax.set_title("Humidity3pm vs RainTomorrow")
                ax.set_xlabel("Rain Tomorrow (No / Yes)")
                ax.set_ylabel("Humidity at 3pm")
                st.pyplot(fig)
            else:
                st.info("`Humidity3pm` feature not available.")

        # 4) Temperature Difference vs RainTomorrow
        elif insight == "Temperature Difference vs RainTomorrow":
            if "TempDiff" in df_ins.columns:
                st.write("**Temperature difference grouped by RainTomorrow:**")
                desc = df_ins.groupby("RainTomorrow")["TempDiff"].describe()
                st.dataframe(desc)

                fig, ax = plt.subplots()
                sns.boxplot(
                    data=df_ins,
                    x="RainTomorrow",
                    y="TempDiff",
                    ax=ax
                )
                ax.set_title("TempDiff vs RainTomorrow")
                ax.set_xlabel("Rain Tomorrow (No / Yes)")
                ax.set_ylabel("Temperature Difference (Max - Min)")
                st.pyplot(fig)
            else:
                st.info("`TempDiff` feature not available.")

        # 5) Top 5 Locations by Rain Probability
        elif insight == "Top 5 Locations by Rain Probability":
            if "Location" in df_ins.columns:
                loc_rain_ins = (
                    df_ins.groupby("Location")["RainFlag"]
                          .mean()
                          .reset_index()
                          .rename(columns={"RainFlag": "RainTomorrowRate"})
                          .sort_values("RainTomorrowRate", ascending=False)
                          .head(5)
                )
                st.write("**Top 5 locations by rain probability:**")
                st.dataframe(loc_rain_ins)

                fig = px.bar(
                    loc_rain_ins,
                    x="Location",
                    y="RainTomorrowRate",
                    title="Top 5 Locations by Rain Probability"
                )
                fig.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("`Location` feature not available.")
