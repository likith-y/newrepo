# app.py
import os
import io
import time
import math
import requests
from bs4 import BeautifulSoup

import streamlit as st
import pandas as pd
import plotly.express as px


try:
    import openai
except Exception:
    openai = None


AIRFARES_FILE = "airfares_index.csv"   
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="AU Airline Demand Insights", layout="wide")
st.title("✈ Airline Booking Market — Demand & Price Insights")

st.markdown(
    """
This app analyzes the local `airfares_index.csv` (you uploaded) and:
- computes price trends (real and nominal),
- highlights routes with notable price changes,
- shows seasonal patterns,
- provides seed popular routes by scraping Wikipedia.
  
> **Assumptions & notes:** This dataset contains fare observations per route per month. There is no passenger-count column; therefore **we infer demand-related signals** from pricing behaviour (price trends, volatility, and relative ranking). If you have passenger volumes, adding them will produce stronger demand signals.
"""
)


@st.cache_data(show_spinner=False)
def read_csv_flexible(path):
    """Read a CSV with encoding / delimiter fallbacks and deduplicate columns."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")

    for enc in ("utf-8", "latin1", "cp1252"):
        for engine in ("c", "python"):
            try:
                df = pd.read_csv(path, encoding=enc, engine=engine, sep=None if engine=="python" else ",")

                cols = pd.Series(df.columns)
                for dup in cols[cols.duplicated()].unique():
                    dup_idx = cols[cols == dup].index.tolist()
                    for i, idx in enumerate(dup_idx):
                        if i != 0:
                            cols[idx] = f"{dup}_{i}"
                df.columns = cols
                return df
            except Exception:
                continue

    df = pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip", quoting=3)
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        dup_idx = cols[cols == dup].index.tolist()
        for i, idx in enumerate(dup_idx):
            if i != 0:
                cols[idx] = f"{dup}_{i}"
    df.columns = cols
    return df

@st.cache_data
def load_airfares_local():
    df = read_csv_flexible(AIRFARES_FILE)
   
   
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("year",):
            colmap[c] = "Year"
        elif cl in ("month",):
            colmap[c] = "Month"
        elif cl in ("yearmonth", "year_month"):
            colmap[c] = "YearMonth"
        elif "port1" in cl or "origin" in cl:
            colmap[c] = "Origin"
        elif "port2" in cl or "destination" in cl:
            colmap[c] = "Destination"
        elif "route" == cl or "route" in cl:
            colmap[c] = "Route"
        elif "$value" in cl or cl in ("value","price"):
            colmap[c] = "Value"
        elif "$real" in cl or cl in ("real","value_real","real_value"):
            colmap[c] = "Real"
    df = df.rename(columns=colmap)

    if "Year" in df.columns and "Month" in df.columns:

        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
        df["date"] = pd.to_datetime(df["Year"].astype(str).str.zfill(4) + "-" + df["Month"].astype(int).astype(str).str.zfill(2) + "-01", errors="coerce")
    elif "YearMonth" in df.columns:
        df["date"] = pd.to_datetime(df["YearMonth"].astype(str) + "01", format="%Y%m%d", errors="coerce")
    else:

        for c in df.columns:
            if "date" in c.lower():
                df["date"] = pd.to_datetime(df[c], errors="coerce")
                break

    if "Real" not in df.columns and "Value" in df.columns:
        df["Real"] = pd.to_numeric(df["Value"], errors="coerce")
    df["Real"] = pd.to_numeric(df["Real"], errors="coerce")

    if "Route" not in df.columns:

        if "Origin" in df.columns and "Destination" in df.columns:
            df["Route"] = df["Origin"].astype(str) + " → " + df["Destination"].astype(str)
        else:
            df["Route"] = df.index.astype(str)
    df = df.dropna(subset=["date"])
    return df

@st.cache_data
def scrape_wiki_busiest_airports(limit=8):
    """Scrape Wikipedia to get top airports to suggest popular routes."""
    url = "https://en.wikipedia.org/wiki/List_of_the_busiest_airports_in_Australia"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", {"class":"wikitable"})
        names = []
        if table:
            for tr in table.select("tr")[1:limit+1]:
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    text = tds[1].get_text(strip=True)

                    names.append(text.split("\n")[0])
        return names
    except Exception:
        return []

#  Load data 
try:
    airfare_df = load_airfares_local()
except FileNotFoundError:
    st.error("Local file `airfares_index.csv` not found in app folder. Please upload it or place it alongside app.py.")
    st.stop()
except Exception as e:
    st.error(f"Failed parsing airfare file: {e}")
    st.stop()

st.sidebar.header("Quick controls")
min_date = airfare_df["date"].min()
max_date = airfare_df["date"].max()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)


top_routes = (airfare_df.groupby("Route")["Real"]
              .median().sort_values().head(200).index.tolist())  # top 200 for selection
selected_routes = st.sidebar.multiselect("Select Routes (pick a few)", options=top_routes, default=top_routes[:6])


seeds = scrape_wiki_busiest_airports()
if seeds:
    st.sidebar.markdown("**Suggested cities (wiki):** " + ", ".join(seeds[:6]))


mask = (airfare_df["date"] >= pd.to_datetime(date_range[0])) & (airfare_df["date"] <= pd.to_datetime(date_range[1]))
if selected_routes:
    mask &= airfare_df["Route"].isin(selected_routes)
filtered = airfare_df[mask].copy()


st.header("Top-line insights")


monthly = filtered.groupby(pd.Grouper(key="date", freq="MS"))["Real"].median().reset_index()
monthly["pct_change"] = monthly["Real"].pct_change() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Date range", f"{date_range[0]} → {date_range[1]}")
c2.metric("Median fare (latest)", f"${monthly['Real'].iloc[-1]:.2f}" if not monthly.empty else "n/a",
          f"{monthly['pct_change'].iloc[-1]:+.2f}% MoM" if len(monthly) > 1 else "")

vol = filtered["Real"].std() / (filtered["Real"].mean() + 1e-9) if not filtered["Real"].empty else float("nan")
c3.metric("Price volatility (CV)", f"{vol:.2f}")

st.markdown("**Notes:** `Real` is CPI-adjusted fare (if present in your file). Where not present, `Value` is used as fallback.")

st.subheader("Top / Bottom routes (by median Real fare)")
route_stats = (filtered.groupby("Route")["Real"]
               .agg(["median","mean","std","count"])
               .rename(columns={"median":"median_real","mean":"mean_real","std":"std_real","count":"n_obs"})
               .sort_values("median_real"))
st.dataframe(pd.concat([route_stats.head(10), route_stats.tail(10)]))

st.subheader("Price trends — route(s) over time")
if filtered.empty:
    st.warning("No data for selected filters.")
else:
    fig = px.line(filtered, x="date", y="Real", color="Route", markers=False,
                  title="Real fare over time for selected routes (monthly samples)")
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Routes with notable price changes (recent 12m vs prior 12m)")

def detect_price_change(df, months=12):
    end = df["date"].max()
    start_recent = end - pd.DateOffset(months=months-1)
    start_prior = end - pd.DateOffset(months=2*months-1)
    recent = df[(df["date"] >= start_recent) & (df["date"] <= end)]
    prior = df[(df["date"] >= start_prior) & (df["date"] < start_recent)]
    stats = []
    for route, g in df.groupby("Route"):
        r_med = recent[recent["Route"]==route]["Real"].median()
        p_med = prior[prior["Route"]==route]["Real"].median()
        if pd.notna(r_med) and pd.notna(p_med) and p_med>0:
            pct = (r_med - p_med) / p_med * 100
            stats.append({"Route":route, "prior_med":p_med, "recent_med":r_med, "pct_change":pct})
    s = pd.DataFrame(stats).sort_values("pct_change", ascending=False)
    return s

change_df = detect_price_change(airfare_df) 
if change_df.empty:
    st.info("Not enough history to compute change scores.")
else:
    st.dataframe(change_df.head(15))

st.subheader("Seasonality (avg real fare by month-of-year)")
filtered["month_of_year"] = filtered["date"].dt.month
season = (filtered.groupby(["Route","month_of_year"])["Real"].median().reset_index()
          .pivot(index="Route", columns="month_of_year", values="Real").fillna(math.nan))

display_routes = selected_routes if selected_routes else route_stats.head(20).index.tolist()
season_display = season.loc[season.index.isin(display_routes)]
if season_display.empty:
    st.info("No seasonality data for selected routes.")
else:
    st.dataframe(season_display)


st.markdown("---")
st.subheader("Auto-generated insights (optional)")

openai_api_key = st.text_input("OpenAI API Key (leave blank to skip)", type="password")
if openai_api_key:
    if openai is None:
        st.error("OpenAI library not installed. Install `openai` in requirements to enable.")
    else:
        openai.api_key = openai_api_key
        if st.button("Generate Insights (GPT)"):

            top_changes = change_df.head(10).to_dict(orient="records")
            prompt = (
                "You are an analyst. Given the following recent price changes for airline routes (prior 12 months vs recent 12 months):\n\n"
                f"{top_changes}\n\n"
                "Provide a concise bullet-list of insights and suggested actions for a hostel group that wants to buy cheaper tickets or design promotions around demand. Keep it short (6 bullets)."
            )
            with st.spinner("Contacting OpenAI..."):
                try:
                    resp = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=400,
                        temperature=0.2,
                    )
                    text = resp.choices[0].message.content.strip()
                    st.markdown(text)
                except Exception as e:
                    st.error(f"OpenAI call failed: {e}")

st.markdown("---")
st.subheader("Export / Download")
export_df = filtered.copy()
export_df = export_df.sort_values("date")
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered data (CSV)", data=csv_bytes, file_name="filtered_airfares.csv", mime="text/csv")

st.success("Done — review the charts and the route stats. Add passenger-volume data for stronger demand signals.")
