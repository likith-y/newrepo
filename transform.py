import pandas as pd
import numpy as np

SCHOOL_HOLIDAYS_AU = {
    # rough typical periods; for demo heatmap (state-by-state differs in reality)
    "Jan": "Summer peak",
    "Apr": "Autumn break",
    "Jul": "Winter break",
    "Sep": "Spring break",
    "Dec": "Summer peak"
}

def monthly_demand_metrics(route_df, airfares_df, origin=None, destination=None):
    df = route_df.copy()
    if origin and destination:
        mask = (df["origin"].str.contains(origin, case=False, na=False)) & \
               (df["destination"].str.contains(destination, case=False, na=False))
        df = df[mask]
    # Aggregate passengers per month for the route
    pax_monthly = df.groupby("date", as_index=False)["passengers"].sum()

    # Airfare index (join on date)
    fares = airfares_df[airfares_df["fare_type"].str.contains("Discount", case=False, na=False)]
    joined = pd.merge(pax_monthly, fares[["date","index"]], on="date", how="left")

    # Trend stats
    if len(joined) >= 3:
        joined["pax_ma3"] = joined["passengers"].rolling(3).mean()
    else:
        joined["pax_ma3"] = joined["passengers"]

    # High-demand months (top quartile)
    q75 = joined["passengers"].quantile(0.75) if not joined["passengers"].empty else np.nan
    joined["high_demand_flag"] = joined["passengers"] >= q75 if not np.isnan(q75) else False
    joined["month_name"] = joined["date"].dt.strftime("%b")
    joined["seasonal_label"] = joined["month_name"].map(SCHOOL_HOLIDAYS_AU).fillna("Normal")

    # Price trend: change vs previous month
    joined["price_mom_%"] = joined["index"].pct_change()*100
    joined["pax_mom_%"]   = joined["passengers"].pct_change()*100

    # Popular routes overall (if no specific filter)
    return joined

def top_routes(route_df, n=10):
    grp = (route_df.groupby(["origin","destination"], as_index=False)["passengers"]
           .sum()
           .sort_values("passengers", ascending=False))
    return grp.head(n)
