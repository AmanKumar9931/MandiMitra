# ================================================================
# MANDIMITRA INDIA
# AI-POWERED AGRICULTURAL MARKET INTELLIGENCE SYSTEM
# ================================================================

import os
import json
import warnings
import re

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ================================================================
# 1. PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="MandiMitra India",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# 2. DIRECTORY CONFIGURATION
# ================================================================

# Works in both a Python script and Jupyter environment.

if "__file__" in globals():
    BASE_DIR = Path(__file__).resolve().parent
else:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

print("Project Directory:", BASE_DIR)


# ================================================================
# 3. FILE PATHS
# ================================================================

CSV_CANDIDATES = [

    BASE_DIR / "All_India_Mandi_2026.csv",

    BASE_DIR / "MandiMitra_Dashboard_Data.csv",

    BASE_DIR / "datasets" / "All_India_Mandi_2026.csv",

    BASE_DIR / "datasets" / "MandiMitra_Dashboard_Data.csv",

    DATA_DIR / "All_India_Mandi_2026.csv",

    DATA_DIR / "MandiMitra_Dashboard_Data.csv",

    DATA_DIR / "district_commodity_prices.csv"

]

GEOJSON_CANDIDATES = [

    DATA_DIR / "india_interactive.geojson",

    BASE_DIR / "india_interactive.geojson",

    BASE_DIR / "maps" / "india_interactive.geojson",

    BASE_DIR / "india_districts.geojson"

]


# ================================================================
# 4. CUSTOM CSS
# ================================================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1020;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background-color: #111a2c;
}

[data-testid="stMetric"] {
    background-color: #172238;
    border: 1px solid #293955;
    padding: 18px;
    border-radius: 14px;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

h1, h2, h3 {
    color: #f8fafc;
}

.stButton button {
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)


# ================================================================
# 5. HELPER FUNCTIONS
# ================================================================

def find_file(paths):

    for path in paths:

        if path.exists():
            return path

    return None


def normalize_name(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).strip().lower()
    )


def money(value):

    if pd.isna(value):
        return "N/A"

    return f"₹{value:,.2f}"


def standardize_columns(df):

    aliases = {

        "State": [
            "state",
            "state_name",
            "st_nm"
        ],

        "District": [
            "district",
            "district_name"
        ],

        "Market": [
            "market",
            "market_name",
            "mandi",
            "apmc"
        ],

        "Commodity": [
            "commodity",
            "commodity_name",
            "crop"
        ],

        "Variety": [
            "variety",
            "variety_name"
        ],

        "Grade": [
            "grade"
        ],

        "Arrival_Date": [
            "arrival_date",
            "date",
            "reported_date"
        ],

        "Min_Price": [
            "min_price",
            "minimum_price"
        ],

        "Max_Price": [
            "max_price",
            "maximum_price"
        ],

        "Modal_Price": [
            "modal_price",
            "model_price",
            "price"
        ],

        "Commodity_Code": [
            "commodity_code"
        ]

    }

    existing = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    rename = {}

    for target, candidates in aliases.items():

        if target not in df.columns:

            for candidate in candidates:

                if candidate in existing:

                    rename[existing[candidate]] = target
                    break

    return df.rename(columns=rename)


# ================================================================
# 6. LOAD DATASET
# ================================================================

@st.cache_data(show_spinner="Loading Indian mandi data...")

def load_data(file_path, modified):

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    df = standardize_columns(df)

    required_columns = [

        "State",
        "District",
        "Market",
        "Commodity",
        "Modal_Price"

    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # Text columns

    text_columns = [

        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade"

    ]

    for col in text_columns:

        if col not in df.columns:

            df[col] = "Unknown"

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Numeric columns

    numeric_columns = [

        "Min_Price",
        "Max_Price",
        "Modal_Price"

    ]

    for col in numeric_columns:

        if col not in df.columns:

            df[col] = np.nan

        df[col] = pd.to_numeric(

            df[col].astype(str).str.replace(
                ",", "", regex=False
            ),

            errors="coerce"

        )

    # Date conversion

    if "Arrival_Date" not in df.columns:

        df["Arrival_Date"] = pd.NaT

    df["Arrival_Date"] = pd.to_datetime(

        df["Arrival_Date"],

        errors="coerce",

        dayfirst=True

    )

    # Remove invalid prices

    df = df.dropna(
        subset=["Modal_Price"]
    )

    df = df[
        df["Modal_Price"] >= 0
    ].copy()

    # Map key

    df["Map_Key"] = (

        df["State"].apply(normalize_name)

        + "_"

        + df["District"].apply(normalize_name)

    )

    return df


# ================================================================
# 7. LOAD GEOJSON
# ================================================================

@st.cache_data(show_spinner=False)

def load_geojson(file_path, modified):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        geo = json.load(f)

    for feature in geo.get("features", []):

        props = feature.setdefault(
            "properties", {}
        )

        state = props.get(
            "st_nm",
            props.get(
                "State",
                props.get("state", "")
            )
        )

        district = props.get(
            "district",
            props.get(
                "District",
                props.get("dt_name", "")
            )
        )

        props["map_key"] = (

            normalize_name(state)

            + "_"

            + normalize_name(district)

        )

    return geo


# ================================================================
# 8. SIDEBAR FILTERS
# ================================================================

def apply_filters(df):

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🔎 Market Filters"
    )

    states = [

        "All India"

    ] + sorted(
        df["State"].dropna().unique().tolist()
    )

    state = st.sidebar.selectbox(
        "Select State",
        states
    )

    filtered = df.copy()

    if state != "All India":

        filtered = filtered[
            filtered["State"] == state
        ]

    districts = [

        "All Districts"

    ] + sorted(
        filtered["District"].unique().tolist()
    )

    district = st.sidebar.selectbox(
        "Select District",
        districts
    )

    if district != "All Districts":

        filtered = filtered[
            filtered["District"] == district
        ]

    markets = [

        "All Mandis"

    ] + sorted(
        filtered["Market"].unique().tolist()
    )

    market = st.sidebar.selectbox(
        "Select Mandi",
        markets
    )

    if market != "All Mandis":

        filtered = filtered[
            filtered["Market"] == market
        ]

    commodities = [

        "All Commodities"

    ] + sorted(
        filtered["Commodity"].unique().tolist()
    )

    commodity = st.sidebar.selectbox(
        "Select Commodity",
        commodities
    )

    if commodity != "All Commodities":

        filtered = filtered[
            filtered["Commodity"] == commodity
        ]

    # Date filter

    valid_dates = df["Arrival_Date"].dropna()

    if not valid_dates.empty:

        date_range = st.sidebar.date_input(

            "Select Date Range",

            value=(

                valid_dates.min().date(),

                valid_dates.max().date()

            )

        )

        if isinstance(date_range, tuple) and len(date_range) == 2:

            filtered = filtered[

                filtered["Arrival_Date"].dt.date.between(

                    date_range[0],

                    date_range[1]

                )

            ]

    return filtered


# ================================================================
# 9. INDIA OVERVIEW
# ================================================================

def india_overview(df):

    st.title(
        "🌾 MandiMitra India"
    )

    st.subheader(
        "AI-Powered Agricultural Market Intelligence"
    )

    st.write(
        "Explore Indian agricultural markets, compare mandi "
        "prices and discover data-driven insights."
    )

    if df.empty:

        st.warning(
            "No data available for selected filters."
        )

        return

    # KPI CARDS

    metrics = [

        ("Total Records", f"{len(df):,}"),

        ("States", f"{df['State'].nunique():,}"),

        ("Districts", f"{df['District'].nunique():,}"),

        ("Commodities", f"{df['Commodity'].nunique():,}"),

        ("Mandis", f"{df['Market'].nunique():,}")

    ]

    cols = st.columns(5)

    for col, (label, value) in zip(
        cols,
        metrics
    ):

        col.metric(
            label,
            value
        )

    st.markdown("---")

    # STATE AND COMMODITY CHARTS

    left, right = st.columns(2)

    with left:

        st.subheader(
            "State-wise Average Modal Price"
        )

        state_avg = df.groupby(
            "State",
            as_index=False
        )["Modal_Price"].mean()

        state_avg = state_avg.nlargest(
            15,
            "Modal_Price"
        )

        fig = px.bar(

            state_avg.sort_values(
                "Modal_Price"
            ),

            x="Modal_Price",

            y="State",

            orientation="h",

            color="Modal_Price",

            color_continuous_scale="Turbo",

            template="plotly_dark"

        )

        fig.update_layout(
            height=450,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Top 15 Commodities by Average Price"
        )

        commodity_avg = df.groupby(
            "Commodity",
            as_index=False
        )["Modal_Price"].mean()

        commodity_avg = commodity_avg.nlargest(
            15,
            "Modal_Price"
        )

        fig = px.bar(

            commodity_avg.sort_values(
                "Modal_Price"
            ),

            x="Modal_Price",

            y="Commodity",

            orientation="h",

            color="Modal_Price",

            color_continuous_scale="Plasma",

            template="plotly_dark"

        )

        fig.update_layout(
            height=450,
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # DAILY TREND

    st.subheader(
        "Daily Agricultural Price Trend"
    )

    dated = df.dropna(
        subset=["Arrival_Date"]
    )

    if not dated.empty:

        trend = dated.groupby(
            "Arrival_Date",
            as_index=False
        )["Modal_Price"].mean()

        fig = px.line(

            trend,

            x="Arrival_Date",

            y="Modal_Price",

            markers=True,

            template="plotly_dark"

        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # TOP MARKETS

    st.subheader(
        "Top 10 Markets by Average Modal Price"
    )

    top_markets = df.groupby(

        ["State", "District", "Market"],

        as_index=False

    ).agg(

        Modal_Price=(
            "Modal_Price",
            "mean"
        ),

        Records=(
            "Modal_Price",
            "size"
        )

    ).nlargest(
        10,
        "Modal_Price"
    )

    st.dataframe(
        top_markets,
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# 10. INTERACTIVE INDIA MAP
# ================================================================

def interactive_map(df, geo):

    st.title(
        "🗺️ Interactive India District Map"
    )

    if geo is None:

        st.warning(
            "District GeoJSON file not found. "
            "Add india_interactive.geojson inside data/."
        )

        return

    commodities = sorted(
        df["Commodity"].unique().tolist()
    )

    if not commodities:

        st.info(
            "No commodity data available."
        )

        return

    col1, col2 = st.columns(2)

    with col1:

        commodity = st.selectbox(
            "Select Commodity",
            commodities,
            key="map_commodity"
        )

    with col2:

        metric = st.selectbox(

            "Select Map Metric",

            [
                "Average_Price",
                "Minimum_Price",
                "Maximum_Price",
                "Records"
            ],

            format_func=lambda x: {

                "Average_Price": "Average Modal Price",

                "Minimum_Price": "Minimum Price",

                "Maximum_Price": "Maximum Price",

                "Records": "Number of Records"

            }[x]

        )

    sub = df[
        df["Commodity"] == commodity
    ]

    summary = sub.groupby(

        ["State", "District", "Map_Key"],

        as_index=False

    ).agg(

        Average_Price=(
            "Modal_Price",
            "mean"
        ),

        Minimum_Price=(
            "Min_Price",
            "min"
        ),

        Maximum_Price=(
            "Max_Price",
            "max"
        ),

        Records=(
            "Modal_Price",
            "size"
        ),

        Mandis=(
            "Market",
            "nunique"
        )

    )

    if summary.empty:

        st.info(
            "No records available."
        )

        return

    fig = px.choropleth(

        summary,

        geojson=geo,

        locations="Map_Key",

        featureidkey="properties.map_key",

        color=metric,

        hover_name="District",

        hover_data=[

            "State",

            "Average_Price",

            "Minimum_Price",

            "Maximum_Price",

            "Records",

            "Mandis"

        ],

        color_continuous_scale="Turbo",

        template="plotly_dark",

        title=f"{commodity} - District-wise Market Intelligence"

    )

    fig.update_geos(

        fitbounds="locations",

        visible=False

    )

    fig.update_layout(

        height=700,

        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "District-wise Price Summary"
    )

    st.dataframe(

        summary.sort_values(
            "Average_Price",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True

    )


# ================================================================
# 11. MARKET EXPLORER
# ================================================================

def market_explorer(df):

    st.title(
        "🔎 Agricultural Market Explorer"
    )

    if df.empty:

        st.info(
            "No records available."
        )

        return

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Average Price",
        money(df["Modal_Price"].mean())
    )

    col3.metric(
        "Total Mandis",
        df["Market"].nunique()
    )

    st.subheader(
        "Price Distribution"
    )

    fig = px.histogram(

        df,

        x="Modal_Price",

        nbins=40,

        template="plotly_dark"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Detailed Market Records"
    )

    columns = [

        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Arrival_Date",
        "Min_Price",
        "Max_Price",
        "Modal_Price"

    ]

    columns = [
        c for c in columns
        if c in df.columns
    ]

    st.dataframe(
        df[columns],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(

        "📥 Download Filtered Dataset",

        df[columns].to_csv(
            index=False
        ).encode("utf-8"),

        file_name="MandiMitra_Filtered_Data.csv",

        mime="text/csv"

    )


# ================================================================
# 12. COMMODITY INTELLIGENCE
# ================================================================

def commodity_intelligence(df):

    st.title(
        "🌽 Commodity Intelligence"
    )

    commodities = sorted(
        df["Commodity"].unique().tolist()
    )

    if not commodities:

        st.info(
            "No commodities available."
        )

        return

    commodity = st.selectbox(
        "Select Commodity",
        commodities,
        key="commodity_intelligence"
    )

    sub = df[
        df["Commodity"] == commodity
    ]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        len(sub)
    )

    c2.metric(
        "Average Price",
        money(sub["Modal_Price"].mean())
    )

    c3.metric(
        "Minimum Price",
        money(sub["Modal_Price"].min())
    )

    c4.metric(
        "Maximum Price",
        money(sub["Modal_Price"].max())
    )

    left, right = st.columns(2)

    with left:

        st.subheader(
            "State-wise Price"
        )

        state_data = sub.groupby(
            "State",
            as_index=False
        )["Modal_Price"].mean()

        fig = px.bar(

            state_data.nlargest(
                15,
                "Modal_Price"
            ),

            x="State",

            y="Modal_Price",

            color="Modal_Price",

            template="plotly_dark"

        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Mandi-wise Price"
        )

        mandi_data = sub.groupby(
            ["State", "District", "Market"],
            as_index=False
        )["Modal_Price"].mean()

        fig = px.bar(

            mandi_data.nlargest(
                15,
                "Modal_Price"
            ),

            x="Market",

            y="Modal_Price",

            color="State",

            hover_data=["District"],

            template="plotly_dark"

        )

        fig.update_layout(
            xaxis_tickangle=-45
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Historical trend

    dated = sub.dropna(
        subset=["Arrival_Date"]
    )

    if not dated.empty:

        st.subheader(
            "Commodity Historical Trend"
        )

        trend = dated.groupby(
            "Arrival_Date",
            as_index=False
        )["Modal_Price"].mean()

        fig = px.line(

            trend,

            x="Arrival_Date",

            y="Modal_Price",

            markers=True,

            template="plotly_dark"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Commodity Market Details"
    )

    st.dataframe(
        mandi_data.sort_values(
            "Modal_Price",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# 13. MANDI COMPARISON
# ================================================================

def mandi_comparison(df):

    st.title(
        "⚖️ Mandi Price Comparison"
    )

    commodities = sorted(
        df["Commodity"].unique().tolist()
    )

    if not commodities:

        st.info(
            "No commodity data available."
        )

        return

    commodity = st.selectbox(
        "Select Commodity",
        commodities,
        key="comparison_commodity"
    )

    sub = df[
        df["Commodity"] == commodity
    ]

    markets = sorted(
        sub["Market"].unique().tolist()
    )

    selected = st.multiselect(

        "Select Mandis",

        markets,

        default=markets[:min(3, len(markets))]

    )

    if not selected:

        st.warning(
            "Select at least one mandi."
        )

        return

    result = sub[
        sub["Market"].isin(selected)
    ].groupby(

        ["State", "District", "Market"],

        as_index=False

    ).agg(

        Average_Price=(
            "Modal_Price",
            "mean"
        ),

        Minimum_Price=(
            "Min_Price",
            "min"
        ),

        Maximum_Price=(
            "Max_Price",
            "max"
        ),

        Records=(
            "Modal_Price",
            "size"
        )

    )

    fig = px.bar(

        result,

        x="Market",

        y="Average_Price",

        color="State",

        hover_data=[

            "District",

            "Minimum_Price",

            "Maximum_Price",

            "Records"

        ],

        template="plotly_dark"

    )

    fig.update_layout(
        height=450,
        xaxis_tickangle=-35
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )


# ================================================================
# 14. PRICE VOLATILITY
# ================================================================

def price_volatility(df):

    st.title(
        "📉 Agricultural Price Volatility"
    )

    result = df.groupby(
        "Commodity"
    )["Modal_Price"].agg(

        ["mean", "std", "count"]

    ).reset_index()

    result.rename(

        columns={

            "mean": "Average_Price",

            "std": "Price_StdDev",

            "count": "Records"

        },

        inplace=True

    )

    result["Volatility_Percentage"] = (

        result["Price_StdDev"]

        / result["Average_Price"].replace(
            0,
            np.nan
        )

    ) * 100

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=["Volatility_Percentage"]
    )

    if result.empty:

        st.info(
            "Insufficient data to calculate volatility."
        )

        return

    left, right = st.columns(2)

    with left:

        st.subheader(
            "Most Volatile Commodities"
        )

        volatile = result.nlargest(
            20,
            "Volatility_Percentage"
        )

        fig = px.bar(

            volatile.sort_values(
                "Volatility_Percentage"
            ),

            x="Volatility_Percentage",

            y="Commodity",

            orientation="h",

            color="Volatility_Percentage",

            color_continuous_scale="Turbo",

            template="plotly_dark"

        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader(
            "Volatility vs Average Price"
        )

        fig = px.scatter(

            result,

            x="Average_Price",

            y="Volatility_Percentage",

            size="Records",

            color="Volatility_Percentage",

            hover_name="Commodity",

            template="plotly_dark"

        )

        fig.update_layout(
            height=550
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Volatility Details"
    )

    st.dataframe(

        result.sort_values(
            "Volatility_Percentage",
            ascending=False
        ),

        use_container_width=True,

        hide_index=True

    )


# ================================================================
# 15. AI PRICE PREDICTOR
# ================================================================

def price_predictor(df):

    st.title("🤖 AI Agricultural Price Predictor")
    st.write(
        "Forecast mandi modal prices over a future date range using the trained CatBoost model."
    )

    model_path = find_file([
        MODELS_DIR / "mandimitra_catboost.cbm",
        MODELS_DIR / "catboost_model.cbm",
        MODELS_DIR / "catboost_model.pkl"
    ])
    feature_path = find_file([
        MODELS_DIR / "model_features.json",
        MODELS_DIR / "mandimitra_features.json",
        MODELS_DIR / "features.json"
    ])
    category_path = find_file([
        MODELS_DIR / "categorical_features.json",
        MODELS_DIR / "mandimitra_categorical_features.json"
    ])

    if model_path is None or feature_path is None:
        st.warning("CatBoost model or its feature JSON was not found in models/.")
        return

    try:
        with open(feature_path, "r", encoding="utf-8") as f:
            features = json.load(f)
        if isinstance(features, dict):
            features = features.get("features", features.get("feature_names", []))

        categories = []
        if category_path:
            with open(category_path, "r", encoding="utf-8") as f:
                categories = json.load(f)
            if isinstance(categories, dict):
                categories = categories.get("categorical_features", categories.get("categorical", []))

        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(model_path))
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return

    commodities = sorted(df["Commodity"].dropna().unique().tolist())
    states = sorted(df["State"].dropna().unique().tolist())
    if not commodities or not states:
        st.warning("Commodity or State values are unavailable in the dataset.")
        return

    commodity = st.selectbox("Commodity", commodities, key="predict_commodity")
    state = st.selectbox("State", states, key="predict_state")
    state_data = df[df["State"] == state]
    districts = sorted(state_data["District"].dropna().unique().tolist())
    if not districts:
        st.warning("No districts available for this state.")
        return
    district = st.selectbox("District", districts, key="predict_district")

    selected_data = state_data[
        (state_data["District"] == district) &
        (state_data["Commodity"] == commodity)
    ]
    if selected_data.empty:
        st.warning("No historical records for this selection.")
        return

    markets = sorted(selected_data["Market"].dropna().unique().tolist())
    market = st.selectbox("Mandi", markets, key="predict_market")
    local = selected_data[selected_data["Market"] == market].copy()
    local["Arrival_Date"] = pd.to_datetime(local["Arrival_Date"], errors="coerce")
    local = local.dropna(subset=["Arrival_Date", "Modal_Price"]).sort_values("Arrival_Date")
    if local.empty:
        st.warning("No usable historical prices for this mandi.")
        return

    latest = local.iloc[-1]
    latest_date = pd.Timestamp(latest["Arrival_Date"]).normalize()
    latest_price = float(latest["Modal_Price"])
    st.metric("Latest Modal Price", money(latest_price), help=f"Latest recorded date: {latest_date.date()}")
    st.caption(f"Historical data available through {latest_date.date()}.")

    st.subheader("Predict Price on a Future Date")
    st.caption(
        "Choose the exact date you want to forecast. MandiMitra recursively "
        "estimates each day between the latest historical record and your target date."
    )
    default_target = (latest_date + pd.Timedelta(days=30)).date()
    target_date = st.date_input(
        "Future prediction date",
        value=default_target,
        min_value=(latest_date + pd.Timedelta(days=1)).date(),
        max_value=(latest_date + pd.Timedelta(days=365)).date(),
        key="future_target_date"
    )

    target_timestamp = pd.Timestamp(target_date).normalize()
    forecast_days = (target_timestamp - latest_date).days
    if forecast_days < 1:
        st.error(f"Choose a date after the latest available record ({latest_date.date()}).")
        return
    if forecast_days > 365:
        st.error("The selected date cannot be more than 365 days after the latest historical record.")
        return

    st.info(
        f"Target date: **{target_timestamp.date()}** · "
        f"Forecast horizon: **{forecast_days} calendar days** from the latest record."
    )
    if forecast_days > 90:
        st.warning(
            "This is a long-range forecast (more than 90 days). "
            "Uncertainty can be substantial because future arrivals, weather, "
            "demand and market conditions are not known."
        )

    if not st.button("🚀 Predict Price for Selected Date", type="primary"):
        return

    # Forecast every calendar day up to the selected target date so lag and
    # rolling features can be updated recursively. The target-date row is
    # highlighted in the result.
    history = local.copy().sort_values("Arrival_Date").reset_index(drop=True)
    forecast_rows = []
    start_date = latest_date + pd.Timedelta(days=1)
    end_date = target_timestamp
    forecast_dates = pd.date_range(start_date, end_date, freq="D")

    # Use historical min/modal and max/modal ratios to estimate unknown
    # future min/max prices; these are not observed future market values.
    valid_ratio = history[
        (history["Modal_Price"] > 0) &
        (history["Min_Price"] > 0) &
        (history["Max_Price"] > 0)
    ].tail(60)
    if len(valid_ratio):
        min_ratio = float((valid_ratio["Min_Price"] / valid_ratio["Modal_Price"]).median())
        max_ratio = float((valid_ratio["Max_Price"] / valid_ratio["Modal_Price"]).median())
    else:
        min_ratio, max_ratio = 0.95, 1.05

    missing_required = [c for c in features if c not in history.columns and c not in {
        "Year", "Month", "Day", "Day_of_Week", "Week_of_Year", "Quarter", "Is_Weekend",
        "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Rolling_Mean_3", "Rolling_Mean_7",
        "Rolling_Mean_14", "Price_Change", "Price_Change_Percentage",
        "Price_Volatility_7", "Price_Volatility_14", "Min_Price", "Max_Price", "Modal_Price"
    }]
    if missing_required:
        st.error("Required model inputs are unavailable: " + ", ".join(missing_required))
        st.info("Check that the feature JSON matches the CatBoost model trained in the notebook.")
        return

    try:
        for forecast_date in forecast_dates:
            prices = history["Modal_Price"].astype(float)
            previous_price = float(prices.iloc[-1])
            row = latest.to_dict()

            row.update({
                "State": state, "District": district, "Market": market,
                "Commodity": commodity, "Arrival_Date": forecast_date,
                "Year": forecast_date.year,
                "Month": forecast_date.month,
                "Day": forecast_date.day,
                "Day_of_Week": forecast_date.dayofweek,
                "Week_of_Year": int(forecast_date.isocalendar().week),
                "Quarter": forecast_date.quarter,
                "Is_Weekend": int(forecast_date.dayofweek >= 5),
                "Min_Price": max(0.0, previous_price * min_ratio),
                "Max_Price": max(0.0, previous_price * max_ratio),
                "Modal_Price": previous_price,
                "Lag_1": previous_price,
                "Lag_2": float(prices.iloc[-2]) if len(prices) >= 2 else previous_price,
                "Lag_3": float(prices.iloc[-3]) if len(prices) >= 3 else previous_price,
                "Lag_7": float(prices.iloc[-7]) if len(prices) >= 7 else previous_price,
                "Rolling_Mean_3": float(prices.tail(3).mean()),
                "Rolling_Mean_7": float(prices.tail(7).mean()),
                "Rolling_Mean_14": float(prices.tail(14).mean()),
                "Price_Change": previous_price - float(prices.iloc[-2]) if len(prices) >= 2 else 0.0,
                "Price_Change_Percentage": ((previous_price / float(prices.iloc[-2])) - 1) * 100 if len(prices) >= 2 and float(prices.iloc[-2]) != 0 else 0.0,
                "Price_Volatility_7": float(prices.tail(7).std()) if len(prices) >= 7 else 0.0,
                "Price_Volatility_14": float(prices.tail(14).std()) if len(prices) >= 14 else 0.0
            })

            # Compatibility with alternative date-feature names, if present
            row.update({"day": forecast_date.day, "month": forecast_date.month,
                        "year": forecast_date.year, "DayOfWeek": forecast_date.dayofweek,
                        "day_of_week": forecast_date.dayofweek})

            input_row = {feature: row.get(feature, np.nan) for feature in features}
            X_future = pd.DataFrame([input_row], columns=features)
            for col in categories:
                if col in X_future.columns:
                    X_future[col] = X_future[col].fillna("Unknown").astype(str)

            predicted_price = float(np.asarray(model.predict(X_future)).reshape(-1)[0])
            predicted_price = max(0.0, predicted_price)
            forecast_rows.append({
                "Forecast_Date": forecast_date,
                "Predicted_Modal_Price": predicted_price,
                "Change_From_Previous_Step": predicted_price - previous_price,
                "Change_Percent": ((predicted_price / previous_price) - 1) * 100 if previous_price else 0.0
            })

            # Append the prediction so subsequent days use recursively
            # updated lags and rolling statistics.
            new_history_row = latest.to_dict()
            new_history_row.update({
                "Arrival_Date": forecast_date,
                "Modal_Price": predicted_price,
                "Min_Price": predicted_price * min_ratio,
                "Max_Price": predicted_price * max_ratio
            })
            history = pd.concat([history, pd.DataFrame([new_history_row])], ignore_index=True)

        forecast_df = pd.DataFrame(forecast_rows)
        target_row = forecast_df.loc[
            pd.to_datetime(forecast_df["Forecast_Date"]).dt.normalize() == target_timestamp
        ].iloc[-1]
        target_prediction = float(target_row["Predicted_Modal_Price"])
        total_change = target_prediction - latest_price
        pct_change = ((target_prediction / latest_price) - 1) * 100 if latest_price else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Recorded Price", money(latest_price))
        c2.metric(
            f"Predicted Price · {target_timestamp.strftime('%d %b %Y')}",
            money(target_prediction),
            delta=f"{pct_change:+.2f}% vs latest"
        )
        c3.metric("Expected Price Change", money(total_change), delta=f"{total_change:+,.2f}")

        st.success(
            f"Estimated modal price for {target_timestamp.strftime('%d %B %Y')}: "
            f"**{money(target_prediction)} per quintal**"
        )

        fig = px.line(
            forecast_df, x="Forecast_Date", y="Predicted_Modal_Price",
            markers=True, title=f"{commodity} Price Forecast — {market}",
            template="plotly_dark"
        )
        fig.add_hline(y=latest_price, line_dash="dash", annotation_text="Latest observed price")
        fig.update_layout(xaxis_title="Forecast Date", yaxis_title="Predicted Modal Price (₹/Quintal)", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Date-wise Forecast Results")
        display_df = forecast_df.copy()
        display_df["Forecast_Date"] = display_df["Forecast_Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Forecast CSV", data=csv,
            file_name=f"MandiMitra_{commodity}_{market}_forecast.csv",
            mime="text/csv"
        )

        st.warning(
            "Research estimate only: the model was trained to predict the next available market observation. "
            "This date-range view recursively applies that model to calendar dates. Future Min/Max prices "
            "are estimated from recent historical ratios, and weekends/mandi holidays are not filtered. "
            "Longer-horizon estimates can accumulate error and are not official mandi quotes."
        )

    except Exception as e:
        st.error(f"Forecast generation failed: {e}")


# ================================================================
# 16. MODEL PERFORMANCE
# ================================================================

def model_performance():

    st.title(
        "📊 Machine Learning Model Performance"
    )

    comparison_paths = [

        MODELS_DIR / "model_comparison.csv",

        DATA_DIR / "model_comparison.csv",

        BASE_DIR / "model_comparison.csv"

    ]

    comparison_file = find_file(
        comparison_paths
    )

    if comparison_file is None:

        st.info(
            "Export your model_comparison DataFrame "
            "from Jupyter as model_comparison.csv "
            "and place it inside models/."
        )

        st.markdown("""
        ### Evaluation Metrics

        **MAE:** Mean Absolute Error.

        **RMSE:** Root Mean Squared Error.

        **R² Score:** Coefficient of Determination.

        Lower MAE and RMSE are better.
        Higher R² is generally better.
        """)

        return

    comparison = pd.read_csv(
        comparison_file
    )

    st.subheader(
        "Model Comparison Table"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    if "Model" in comparison.columns:

        metrics = [
            c for c in comparison.columns
            if c in ["MAE", "RMSE", "R2 Score", "R2"]
        ]

        for metric in metrics:

            st.subheader(
                f"{metric} Comparison"
            )

            fig = px.bar(

                comparison,

                x="Model",

                y=metric,

                color="Model",

                template="plotly_dark"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ================================================================
# 17. APPLICATION HEADER
# ================================================================

st.sidebar.title(
    "🌾 MandiMitra"
)

st.sidebar.caption(
    "India Agricultural Intelligence"
)


# ================================================================
# 18. LOAD MAIN DATASET
# ================================================================

csv_path = find_file(
    CSV_CANDIDATES
)

if csv_path is None:

    st.error(
        "All India CSV dataset not found."
    )

    st.write(
        "Place All_India_Mandi_2026.csv "
        "inside your project folder or datasets/."
    )

    st.code("""
MandiMitra/
│
├── app.py
├── All_India_Mandi_2026.csv
│
├── data/
│   └── india_interactive.geojson
│
└── models/
    ├── mandimitra_catboost.cbm
    ├── mandimitra_features.json
    └── categorical_features.json
""")

    st.stop()


try:

    df = load_data(

        str(csv_path),

        csv_path.stat().st_mtime

    )

except Exception as e:

    st.error(
        f"Dataset Error: {e}"
    )

    st.stop()


if df.empty:

    st.error(
        "No valid price records found."
    )

    st.stop()


st.sidebar.success(
    f"Loaded: {csv_path.name}"
)

st.sidebar.caption(
    f"{len(df):,} valid records"
)


# ================================================================
# 19. APPLY SIDEBAR FILTERS
# ================================================================

filtered_df = apply_filters(
    df
)


# ================================================================
# 20. LOAD INDIA GEOJSON
# ================================================================

geo_path = find_file(
    GEOJSON_CANDIDATES
)

geo = None

if geo_path:

    try:

        geo = load_geojson(

            str(geo_path),

            geo_path.stat().st_mtime

        )

    except Exception as e:

        st.sidebar.warning(
            f"Map loading issue: {e}"
        )


# ================================================================
# 21. NAVIGATION
# ================================================================

pages = [

    "India Overview",

    "Interactive India Map",

    "Market Explorer",

    "Commodity Intelligence",

    "AI Price Predictor",

    "Mandi Comparison",

    "Price Volatility",

    "Model Performance"

]

page = st.sidebar.radio(
    "NAVIGATION",
    pages
)


# ================================================================
# 22. PAGE ROUTING
# ================================================================

if page == "India Overview":

    india_overview(
        filtered_df
    )

elif page == "Interactive India Map":

    interactive_map(
        filtered_df,
        geo
    )

elif page == "Market Explorer":

    market_explorer(
        filtered_df
    )

elif page == "Commodity Intelligence":

    commodity_intelligence(
        filtered_df
    )

elif page == "AI Price Predictor":

    price_predictor(
        df
    )

elif page == "Mandi Comparison":

    mandi_comparison(
        filtered_df
    )

elif page == "Price Volatility":

    price_volatility(
        filtered_df
    )

elif page == "Model Performance":

    model_performance()


# ================================================================
# 23. FOOTER
# ================================================================

st.markdown("---")

st.markdown("""

<div style="
    text-align:center;
    padding:20px;
    color:#94a3b8;
">

<h2>🌾 MandiMitra India</h2>

<p>AI-Powered Agricultural Market Intelligence</p>

<p>
Built using Python | Pandas | Streamlit |
Plotly | CatBoost | Machine Learning
</p>

<small>
Agricultural price analytics and predictive intelligence.
</small>

</div>

""", unsafe_allow_html=True)