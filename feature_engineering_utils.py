"""
Extended bureau.csv feature engineering utilities.

This module builds on the rolling aggregation functions to compute
**core portfolio-level features** per SK_ID_CURR as brainstormed.

Core feature themes implemented:
1. Credit Activity Profile
2. Credit Volume & Exposure
3. Overdue & Delinquency
4. Credit Duration & Behavior
5. Credit Dynamics
6. Installments & Affordability
7. Risk Ratios & Flags
8. Temporal Cohorts (simple recency features)

Each function returns a DataFrame [SK_ID_CURR, feature_value].
You can merge them into your main feature set.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from typing import Callable, Optional, Literal

# ------------------------- utility helpers ------------------------- #

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a.astype(float) / b.replace({0: np.nan})).replace([np.inf, -np.inf], np.nan)

def _nan_percentile(x: pd.Series | np.ndarray, q: float) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.nan
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, q))

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Ensure required columns exist
    for col in ["DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"]:
        if col not in d.columns:
            d[col] = np.nan
    d["duration_days"] = d["DAYS_CREDIT_ENDDATE"] - d["DAYS_CREDIT"]
    d["rev_duration_days"] = d["DAYS_CREDIT"] - d["DAYS_CREDIT_ENDDATE"]
    # groupwise lag(1) of rev_duration_days ordered by recency (max DAYS_CREDIT first)
    d = d.sort_values(["SK_ID_CURR", "DAYS_CREDIT"], ascending=[True, False])
    d["rev_duration_days_lag1"] = d.groupby("SK_ID_CURR")["rev_duration_days"].shift(1)
    return d

def f_num_new(df: pd.DataFrame, *, window_days: int = 365) -> pd.DataFrame:
    m = df["DAYS_CREDIT"].between(-int(window_days), 0)
    out = df.loc[m].groupby("SK_ID_CURR").size()
    return out.rename(f"num_new_w{window_days}").reset_index()

def _normalize_bureau_cats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "CREDIT_ACTIVE" in d:
        d["CREDIT_ACTIVE"] = d["CREDIT_ACTIVE"].astype(str).str.strip().str.lower()
    if "CREDIT_TYPE" in d:
        d["CREDIT_TYPE"] = d["CREDIT_TYPE"].astype(str).str.strip().str.lower()
    if "CREDIT_CURRENCY" in d:
        d["CREDIT_CURRENCY"] = d["CREDIT_CURRENCY"].astype(str).str.strip().str.lower()
    return d


# ---------------------- core feature builders ---------------------- #

def core_credit_activity(df: pd.DataFrame) -> pd.DataFrame:
    d = _normalize_bureau_cats(df)
    grp = d.groupby("SK_ID_CURR")
    n_active = grp["CREDIT_ACTIVE"].apply(lambda s: (s == "active").sum())
    n_closed = grp["CREDIT_ACTIVE"].apply(lambda s: (s == "closed").sum())
    n_total = grp.size()
    share_active = _safe_div(n_active, n_total)
    n_types = grp["CREDIT_TYPE"].nunique()
    return pd.DataFrame({
        "SK_ID_CURR": n_active.index,
        "n_active_loans": n_active.values,
        "n_closed_loans": n_closed.values,
        "share_active": share_active.values,
        "n_credit_types": n_types.values,
    })


def core_credit_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Total exposure, debt, utilization, concentration."""
    grp = df.groupby("SK_ID_CURR")
    total_sum = grp["AMT_CREDIT_SUM"].sum()
    total_debt = grp["AMT_CREDIT_SUM_DEBT"].sum()
    utilization = _safe_div(total_debt, total_sum)
    max_exposure = grp["AMT_CREDIT_SUM"].max()
    conc_ratio = _safe_div(max_exposure, total_sum)
    return pd.DataFrame({
        "SK_ID_CURR": total_sum.index,
        "total_credit_sum": total_sum.values,
        "total_debt": total_debt.values,
        "utilization_portfolio": utilization.values,
        "max_exposure": max_exposure.values,
        "concentration_ratio": conc_ratio.values,
    })


def core_overdue(df: pd.DataFrame) -> pd.DataFrame:
    """Overdue indicators and magnitudes."""
    grp = df.groupby("SK_ID_CURR")
    ever_overdue = grp["CREDIT_DAY_OVERDUE"].max() > 0
    max_overdue_days = grp["CREDIT_DAY_OVERDUE"].max()
    max_overdue_amt = grp["AMT_CREDIT_MAX_OVERDUE"].max()
    share_over90 = grp["CREDIT_DAY_OVERDUE"].apply(lambda s: (s > 90).mean())
    return pd.DataFrame({
        "SK_ID_CURR": max_overdue_days.index,
        "ever_overdue": ever_overdue.astype(float).values,
        "max_overdue_days": max_overdue_days.values,
        "max_overdue_amt": max_overdue_amt.values,
        "share_loans_over90d": share_over90.values,
    })


def core_duration_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """Loan age, remaining duration, closure profile."""
    grp = df.groupby("SK_ID_CURR")
    avg_age = grp["DAYS_CREDIT"].mean()
    avg_remaining = grp["DAYS_CREDIT_ENDDATE"].mean()
    avg_closed_duration = grp["DAYS_ENDDATE_FACT"].mean()
    recency_last_update = grp["DAYS_CREDIT_UPDATE"].min()
    return pd.DataFrame({
        "SK_ID_CURR": avg_age.index,
        "avg_days_credit": avg_age.values,
        "avg_days_remaining": avg_remaining.values,
        "avg_days_closed": avg_closed_duration.values,
        "recency_last_update": recency_last_update.values,
    })


def core_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """Prolongations and new loan trends."""
    grp = df.groupby("SK_ID_CURR")
    total_prolong = grp["CNT_CREDIT_PROLONG"].sum()
    avg_prolong = grp["CNT_CREDIT_PROLONG"].mean()
    n_recent_2y = grp.apply(lambda g: (g["DAYS_CREDIT"] >= -730).sum())
    n_recent_1y = grp.apply(lambda g: (g["DAYS_CREDIT"] >= -365).sum())
    return pd.DataFrame({
        "SK_ID_CURR": total_prolong.index,
        "total_prolong": total_prolong.values,
        "avg_prolong": avg_prolong.values,
        "n_loans_2y": n_recent_2y.values,
        "n_loans_1y": n_recent_1y.values,
    })


def core_installments(df: pd.DataFrame) -> pd.DataFrame:
    """Installment burden vs debt/exposure."""
    grp = df.groupby("SK_ID_CURR")
    total_annuity = grp["AMT_ANNUITY"].sum()
    total_debt = grp["AMT_CREDIT_SUM_DEBT"].sum()
    total_sum = grp["AMT_CREDIT_SUM"].sum()
    annuity_over_debt = _safe_div(total_annuity, total_debt)
    annuity_over_sum = _safe_div(total_annuity, total_sum)
    return pd.DataFrame({
        "SK_ID_CURR": total_annuity.index,
        "total_annuity": total_annuity.values,
        "annuity_over_debt": annuity_over_debt.values,
        "annuity_over_sum": annuity_over_sum.values,
    })


def core_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Risky flags: high utilization, chronic delinquent."""
    grp = df.groupby("SK_ID_CURR")
    high_util = grp.apply(lambda g: any(_safe_div(g["AMT_CREDIT_SUM_DEBT"], g["AMT_CREDIT_SUM"]) > 0.9))
    chronic_delinq = grp["CREDIT_DAY_OVERDUE"].apply(lambda s: (s > 365).any())
    return pd.DataFrame({
        "SK_ID_CURR": high_util.index,
        "flag_high_util": high_util.astype(float).values,
        "flag_chronic_delinq": chronic_delinq.astype(float).values,
    })


def core_temporal(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["weight"] = np.exp(d["DAYS_CREDIT"] / 365.0).clip(lower=0)
    # recency-weighted debt
    def _rwd(g: pd.DataFrame) -> float:
        v = g["AMT_CREDIT_SUM_DEBT"].astype(float).fillna(0).to_numpy()
        w = g["weight"].astype(float).to_numpy()
        den = w.sum()
        return np.nan if den == 0 else float((v * w).sum() / den)

    # last loan by recency (max DAYS_CREDIT ≈ closest to 0)
    idx_last = d.groupby("SK_ID_CURR")["DAYS_CREDIT"].idxmax()
    last = d.loc[idx_last, ["SK_ID_CURR", "AMT_CREDIT_SUM_DEBT", "CREDIT_TYPE"]]
    out = d.groupby("SK_ID_CURR").apply(_rwd).rename("recency_weighted_debt").reset_index()
    out = out.merge(last.rename(columns={
        "AMT_CREDIT_SUM_DEBT": "lastloan_debt",
        "CREDIT_TYPE": "lastloan_type"
    }), on="SK_ID_CURR", how="left")
    return out


# -------------------------- feature factory ------------------------ #

def build_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all core features and merge into one DataFrame."""
    dfs = [
        core_credit_activity(df),
        core_credit_volume(df),
        core_overdue(df),
        core_duration_behavior(df),
        core_dynamics(df),
        core_installments(df),
        core_flags(df),
        core_temporal(df),
    ]
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on="SK_ID_CURR", how="left")
    return out


# if __name__ == "__main__":
#     # minimal smoke test
#     data = {
#         "SK_ID_CURR": [1,1,1,2,2],
#         "CREDIT_ACTIVE": ["Active","Closed","Active","Active","Closed"],
#         "CREDIT_TYPE": ["Credit card","Consumer credit","Credit card","Credit card","Mortgage"],
#         "DAYS_CREDIT": [-30,-200,-500,-100,-400],
#         "DAYS_CREDIT_ENDDATE": [200,100,-50,50,-10],
#         "DAYS_ENDDATE_FACT": [np.nan,-20,-300,np.nan,-50],
#         "DAYS_CREDIT_UPDATE": [-5,-100,-200,-30,-500],
#         "CREDIT_DAY_OVERDUE": [0,120,0,0,400],
#         "AMT_CREDIT_SUM": [1000,2000,500,1500,3000],
#         "AMT_CREDIT_SUM_DEBT": [500,0,100,800,1000],
#         "AMT_CREDIT_MAX_OVERDUE": [0,300,0,0,800],
#         "CNT_CREDIT_PROLONG": [0,1,0,2,0],
#         "AMT_ANNUITY": [50,0,10,60,90],
#     }
#     df = pd.DataFrame(data)
#     feats = build_core_features(df)
#     print(feats.head())


# ---------------------------- core feature set ---------------------------- #
# Aggregate the broader, non-rolling, per-customer features we brainstormed.

from math import exp

def _shannon_entropy(proportions: pd.Series) -> float:
    p = proportions[proportions > 0].astype(float)
    if p.empty:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _within_window(df: pd.DataFrame, window_days: int, time_col: str = "DAYS_CREDIT") -> pd.DataFrame:
    return df[(df[time_col] >= -int(window_days)) & (df[time_col] <= 0)]


def _recent_oldest_rows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return most recent (max DAYS_CREDIT) and oldest (min DAYS_CREDIT) rows per SK_ID_CURR."""
    idx_recent = df.groupby("SK_ID_CURR")["DAYS_CREDIT"].idxmax()
    idx_oldest = df.groupby("SK_ID_CURR")["DAYS_CREDIT"].idxmin()
    return df.loc[idx_recent], df.loc[idx_oldest]


def compute_bureau_features(
    df: pd.DataFrame,
    *,
    windows: list[int] | None = None,
    credit_card_label: str = "credit card",
    risky_types: tuple[str, ...] = ("microloan", "cash loans", "cash loan", "micro loan"),
    time_col: str = "DAYS_CREDIT",
) -> pd.DataFrame:
    """
    Compute the broader feature set per SK_ID_CURR. Produces:
      - "all_" features across entire history
      - Windowed variants (w{days}_) for each value in `windows`

    Parameters
    ----------
    df : bureau-like DataFrame
    windows : list of lookback windows in days (e.g., [365, 730]); if None, uses []
    credit_card_label : lowercase label that matches CREDIT_TYPE for credit cards
    risky_types : iterable of lowercase labels considered 'risky'
    time_col : days before application column (negative values)

    Returns
    -------
    DataFrame with one row per SK_ID_CURR containing engineered features.
    """
    if windows is None:
        windows = []

    d = add_derived_columns(df)
    d["__ctype"] = d["CREDIT_TYPE"].astype(str).str.strip().str.lower()
    d["__active"] = d["CREDIT_ACTIVE"].astype(str).str.strip().str.lower().eq("active")
    d["__closed"] = d["CREDIT_ACTIVE"].astype(str).str.strip().str.lower().eq("closed")
    d["__is_cc"] = d["__ctype"].eq(credit_card_label)

    # Helper to compute a block of features for a given slice
    def block(feat_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if feat_df.empty:
            # Return empty shell with just SK_ID_CURR to keep merges consistent
            return pd.DataFrame({"SK_ID_CURR": d["SK_ID_CURR"].unique()})

        grp = feat_df.groupby("SK_ID_CURR")

        # Activity profile
        n_active = grp["__active"].sum().rename(f"{prefix}n_active_loans")
        n_closed = grp["__closed"].sum().rename(f"{prefix}n_closed_loans")
        n_total = grp.size().rename(f"{prefix}n_total_loans")
        active_ratio = (n_active / n_total.replace({0: np.nan})).rename(f"{prefix}active_ratio")

        # Type & currency diversity
        type_counts = feat_df.pivot_table(index="SK_ID_CURR", columns="__ctype", values="SK_BUREAU_ID", aggfunc="count", fill_value=0)
        type_share = type_counts.div(type_counts.sum(axis=1).replace(0, np.nan), axis=0)
        diversity_index = type_share.apply(_shannon_entropy, axis=1).rename(f"{prefix}type_diversity_shannon")
        currency_diversity = grp["CREDIT_CURRENCY"].nunique().rename(f"{prefix}currency_diversity")

        # Volume & exposure
        total_debt = grp["AMT_CREDIT_SUM_DEBT"].sum().rename(f"{prefix}total_debt")
        total_expo = grp["AMT_CREDIT_SUM"].sum().rename(f"{prefix}total_exposure")
        utilization_port = (total_debt / total_expo.replace({0: np.nan})).rename(f"{prefix}utilization_portfolio")
        # concentration: max single-loan exposure share
        max_expo = grp["AMT_CREDIT_SUM"].max()
        concentration = (max_expo / total_expo.replace({0: np.nan})).rename(f"{prefix}expo_concentration")

        # Overdue & delinquency
        ever_overdue = grp["CREDIT_DAY_OVERDUE"].apply(lambda s: (s.fillna(0) > 0).any()).astype(float).rename(f"{prefix}ever_overdue")
        max_overdue_days = grp["CREDIT_DAY_OVERDUE"].max().rename(f"{prefix}max_overdue_days")
        max_overdue_amt = grp["AMT_CREDIT_MAX_OVERDUE"].max().rename(f"{prefix}max_overdue_amt")
        share_over90 = grp["CREDIT_DAY_OVERDUE"].apply(lambda s: (s.fillna(0) > 90).mean()).rename(f"{prefix}share_overdue_gt90")
        current_over_amt = grp["AMT_CREDIT_SUM_OVERDUE"].sum().rename(f"{prefix}current_overdue_amt")
        avg_overdue_active = (
            feat_df.loc[feat_df["__active"], :]
                   .groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].mean()
                   .rename(f"{prefix}avg_overdue_active")
        )

        # Duration & behavior
        avg_age = grp[time_col].mean().rename(f"{prefix}avg_days_credit")
        avg_remaining = grp["DAYS_CREDIT_ENDDATE"].mean().rename(f"{prefix}avg_days_remaining")
        avg_closed_duration = (
            feat_df.loc[feat_df["__closed"], :]
                   .assign(_duration=lambda x: -(x["DAYS_ENDDATE_FACT"].astype(float)))
                   .groupby("SK_ID_CURR")["_duration"].mean()
                   .rename(f"{prefix}avg_closed_duration")
        )
        duration_var = grp["duration_days"].var(ddof=0).rename(f"{prefix}duration_var")
        last_update_recency = grp["DAYS_CREDIT_UPDATE"].min().rename(f"{prefix}last_update_min_days")

        # Dynamics
        # Exposure growth: compare most recent vs oldest credit sum
        recent_rows, oldest_rows = _recent_oldest_rows(feat_df)
        growth = (
            (recent_rows.set_index("SK_ID_CURR")["AMT_CREDIT_SUM"] - oldest_rows.set_index("SK_ID_CURR")["AMT_CREDIT_SUM"]).rename(f"{prefix}expo_growth_abs")
        )
        growth_rate = _safe_div(
            recent_rows.set_index("SK_ID_CURR")["AMT_CREDIT_SUM"] - oldest_rows.set_index("SK_ID_CURR")["AMT_CREDIT_SUM"],
            oldest_rows.set_index("SK_ID_CURR")["AMT_CREDIT_SUM"],
        ).rename(f"{prefix}expo_growth_rate")

        # Installments & affordability
        total_annuity = grp["AMT_ANNUITY"].sum().rename(f"{prefix}total_annuity")
        annuity_to_debt = _safe_div(total_annuity, total_debt).rename(f"{prefix}annuity_over_total_debt")

        # Risk flags
        high_util_card = (
            feat_df.loc[feat_df["__is_cc"]]
                   .assign(_hu=lambda x: _safe_div(x["AMT_CREDIT_SUM_DEBT"], x["AMT_CREDIT_SUM_LIMIT"]) > 0.9)
                   .groupby("SK_ID_CURR")["_hu"].max().fillna(False).astype(float)
                   .rename(f"{prefix}high_util_card_any")
        )
        chronic_delinquent = grp["CREDIT_DAY_OVERDUE"].apply(lambda s: (s.fillna(0) > 365).any()).astype(float).rename(f"{prefix}chronic_delinquent")
        # short-term risk: ending within 365d & active & has overdue now
        short_term_risk = (
            feat_df.assign(_end_soon=lambda x: x["DAYS_CREDIT_ENDDATE"].fillna(0) <= 365)
                   .assign(_ovd_now=lambda x: x["AMT_CREDIT_SUM_OVERDUE"].fillna(0) > 0)
                   .assign(_flag=lambda x: x["_end_soon"] & x["__active"] & x["_ovd_now"]) 
                   .groupby("SK_ID_CURR")["_flag"].max().astype(float)
                   .rename(f"{prefix}short_term_risk")
        )

        # Temporal cohorts
        # Recency-weighted debt: weight = exp(DAYS_CREDIT / tau) with tau=365
        tau = 365.0
        rwd = (
            feat_df.assign(_w=lambda x: np.exp(x[time_col] / tau))
                   .assign(_wd=lambda x: x["AMT_CREDIT_SUM_DEBT"].fillna(0) * x["_w"]) 
                   .groupby("SK_ID_CURR")["_wd"].sum()
                   .rename(f"{prefix}recency_weighted_debt_tau365")
        )

        # Last-loan characteristics
        recent_rows2, _ = _recent_oldest_rows(feat_df)
        last_rows = recent_rows2.set_index("SK_ID_CURR")
        last_debt = last_rows["AMT_CREDIT_SUM_DEBT"].rename(f"{prefix}last_loan_debt")
        last_overdue = last_rows["AMT_CREDIT_SUM_OVERDUE"].rename(f"{prefix}last_loan_overdue_amt")
        last_type = last_rows["__ctype"].rename(f"{prefix}last_loan_type")

        # Creative features
        # Credit concentration among active debt only
        act = feat_df.loc[feat_df["__active"]]
        conc_active = (
            act.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].apply(lambda s: s.max() / s.sum() if s.sum() else np.nan)
               .rename(f"{prefix}debt_concentration_active")
        )
        # Laddering (refinance proxy): enddate earlier than last info update
        ladder_share = (
            feat_df.assign(_lad=lambda x: (x["DAYS_CREDIT_ENDDATE"].astype(float) < x["DAYS_CREDIT_UPDATE"].astype(float)))
                   .groupby("SK_ID_CURR")["_lad"].mean()
                   .rename(f"{prefix}laddering_share")
        )
        # Portfolio churn: closed in last 2y vs active count
        closed_last2y = _within_window(feat_df, 730)["__closed"].groupby(feat_df["SK_ID_CURR"]).sum()
        churn = _safe_div(closed_last2y, n_active).rename(f"{prefix}portfolio_churn_clos2y_over_active")
        # Risky type intensity
        risky_set = {rt.lower() for rt in risky_types}
        risky_share = (
            feat_df.assign(_risky=lambda x: x["__ctype"].isin(risky_set))
                   .groupby("SK_ID_CURR")["_risky"].mean()
                   .rename(f"{prefix}risky_type_share")
        )
        # Prolongation stress (amount-weighted)
        prol_stress = (
            feat_df.assign(_w=lambda x: x["AMT_CREDIT_SUM"].fillna(0))
                   .assign(_ws=lambda x: x["CNT_CREDIT_PROLONG"].fillna(0) * x["_w"]) 
                   .groupby("SK_ID_CURR")["_ws"].sum() / feat_df.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum().replace(0, np.nan)
        ).rename(f"{prefix}prolongation_stress")

        # Assemble
        pieces = [
            n_active, n_closed, n_total, active_ratio,
            diversity_index, currency_diversity,
            total_debt, total_expo, utilization_port, concentration,
            ever_overdue, max_overdue_days, max_overdue_amt, share_over90, current_over_amt, avg_overdue_active,
            avg_age, avg_remaining, avg_closed_duration, duration_var, last_update_recency,
            growth, growth_rate,
            total_annuity, annuity_to_debt,
            high_util_card, chronic_delinquent, short_term_risk,
            rwd,
            last_debt, last_overdue, last_type,
            conc_active, ladder_share, churn, risky_share, prol_stress,
        ]
        out = pd.concat(pieces, axis=1)
        out.reset_index(inplace=True)
        return out

    # # All-history block
    # all_block = block(d, prefix="all_")

    # Windowed blocks (plus simple count-of-new-loans per window)
    blocks = []
    for w in windows:
        dw = _within_window(d, w, time_col=time_col)
        wb = block(dw, prefix=f"w{w}_")
        # num_new_loans in window
        num_new = f_num_new(d, window_days=w).rename(columns={f"num_new_w{w}": f"w{w}_num_new"})
        wb = wb.merge(num_new, on="SK_ID_CURR", how="left")
        # fraction of loans originated in window (vintage profile)
        total_loans = d.groupby("SK_ID_CURR").size().rename("__tot")
        frac_vintage = _safe_div(num_new.set_index("SK_ID_CURR")[f"w{w}_num_new"], total_loans).rename(f"w{w}_vintage_fraction")
        wb = wb.merge(frac_vintage.reset_index(), on="SK_ID_CURR", how="left")
        blocks.append(wb)

    # Merge all
    out = pd.DataFrame({"SK_ID_CURR": d["SK_ID_CURR"].unique()})
    for b in blocks:
        out = out.merge(b, on="SK_ID_CURR", how="left")

    return out


# # ---------------------------- examples ---------------------------- #
# if __name__ == "__main__":
#     # Minimal smoke test with synthetic rows
#     data = {
#         "SK_ID_CURR": [1,1,1,2,2],
#         "SK_BUREAU_ID": [10,11,12,20,21],
#         "DAYS_CREDIT": [-30,-200,-500,-100,-400],
#         "DAYS_CREDIT_ENDDATE": [200,100,-50,50,-10],
#         "DAYS_CREDIT_UPDATE": [-10,-150,-450,-80,-390],
#         "CREDIT_ACTIVE": ["Active","Closed","Active","Active","Closed"],
#         "CREDIT_TYPE": ["Credit card","Consumer credit","Credit card","Credit card","Mortgage"],
#         "CREDIT_CURRENCY": ["RUB","RUB","USD","RUB","RUB"],
#         "CREDIT_DAY_OVERDUE": [0,10,0,5,500],
#         "AMT_CREDIT_SUM": [1000,2000,500,1500,3000],
#         "AMT_CREDIT_SUM_DEBT": [500,0,100,800,1000],
#         "AMT_CREDIT_SUM_LIMIT": [2000,0,1000,2500,0],
#         "AMT_CREDIT_MAX_OVERDUE": [100,0,50,25,700],
#         "AMT_CREDIT_SUM_OVERDUE": [0,0,0,10,300],
#         "AMT_ANNUITY": [50,0,10,60,90],
#     }
#     df = pd.DataFrame(data)

#     features_all = compute_core_features(df, windows=[365, 730])
#     print(features_all.head())


# =========================== bureau_balance features =========================== #
# Functions to join bureau_balance.csv with bureau.csv and engineer features.

BB_STATUS_ORDER = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": -1, "X": np.nan}


def prep_bureau_balance(bb: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize bureau_balance:
      - Keep [SK_BUREAU_ID, MONTHS_BALANCE, STATUS]
      - Coerce types; uppercase STATUS; map to numeric delinquency code (0..5), with C=-1, X=NaN
      - MONTHS_BALANCE: integers where -1 is most recent month relative to application
    """
    cols = ["SK_BUREAU_ID", "MONTHS_BALANCE", "STATUS"]
    bb = bb[cols].copy()
    bb["MONTHS_BALANCE"] = bb["MONTHS_BALANCE"].astype(int)
    bb["STATUS"] = bb["STATUS"].astype(str).str.strip().str.upper()
    bb["STATUS_CODE"] = bb["STATUS"].map(BB_STATUS_ORDER)
    return bb


def _bb_within_window(bb: pd.DataFrame, months: int) -> pd.DataFrame:
    """Filter to last `months` months relative to application: -months <= MB <= -1."""
    return bb[(bb["MONTHS_BALANCE"] >= -int(months)) & (bb["MONTHS_BALANCE"] <= -1)]


def _run_length_encode(flags: pd.Series) -> int:
    """Longest consecutive 1s in a boolean Series (order by MONTHS_BALANCE ascending toward -1)."""
    if flags.empty:
        return 0
    # ensure order from oldest to most recent
    longest = cur = 0
    for v in flags.astype(bool).tolist():
        if v:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return int(longest)


def bb_features_per_account(bb: pd.DataFrame, *, months_windows: list[int] = [6, 12, 24]) -> pd.DataFrame:
    """
    Compute features per SK_BUREAU_ID from bureau_balance.

    For each window w in months_windows, produce prefixed metrics:
      - w{w}_months_count: number of reported months
      - w{w}_dpd_mean / dpd_max / dpd_sum (using STATUS_CODE; C=-1 ignored in mean/sum)
      - w{w}_share_dpd_gt0: share of months with dpd>0
      - w{w}_share_unknown: share of months with STATUS=='X'
      - w{w}_share_closed: share of months with STATUS=='C'
      - w{w}_months_since_last_dpd_gt0: distance (in months) from last month with dpd>0 to -1 (NaN if never)
      - w{w}_longest_streak_dpd_gt0: longest consecutive delinquent streak within window
      - w{w}_status_0..5: fraction of months in each bucket
    Also compute non-windowed recency signals:
      - last_month (min MONTHS_BALANCE, typically -1 if exists)
      - last_status_code, last_status_is_closed
    """
    bb = prep_bureau_balance(bb)

    # last-month signals per account
    idx_last = bb.groupby("SK_BUREAU_ID")["MONTHS_BALANCE"].idxmax()
    last_rows = bb.loc[idx_last]
    base = pd.DataFrame({
        "SK_BUREAU_ID": last_rows["SK_BUREAU_ID"].values,
        "bb_last_month": last_rows["MONTHS_BALANCE"].values,
        "bb_last_status_code": last_rows["STATUS_CODE"].values,
        "bb_last_status_is_closed": (last_rows["STATUS"] == "C").astype(float).values,
    })

    outs = [base]
    for w in months_windows:
        bw = _bb_within_window(bb, w)
        if bw.empty:
            continue
        grp = bw.groupby("SK_BUREAU_ID")
        months_count = grp.size().rename(f"w{w}_months_count")
        # Exclude C(-1) and NaN from mean/sum; use only >=0 codes
        valid = bw[bw["STATUS_CODE"].fillna(-99) >= 0]
        grp_valid = valid.groupby("SK_BUREAU_ID")
        dpd_mean = grp_valid["STATUS_CODE"].mean().rename(f"w{w}_dpd_mean")
        dpd_max = grp_valid["STATUS_CODE"].max().rename(f"w{w}_dpd_max")
        dpd_sum = grp_valid["STATUS_CODE"].sum().rename(f"w{w}_dpd_sum")

        share_dpd_gt0 = grp.apply(lambda g: (g["STATUS_CODE"].fillna(-1) > 0).mean()).rename(f"w{w}_share_dpd_gt0")
        share_dpd_gt1 = grp.apply(lambda g: (g["STATUS_CODE"].fillna(-1) > 1).mean()).rename(f"w{w}_share_dpd_gt1")
        share_dpd_gt2 = grp.apply(lambda g: (g["STATUS_CODE"].fillna(-1) > 2).mean()).rename(f"w{w}_share_dpd_gt2")
        share_dpd_gt3 = grp.apply(lambda g: (g["STATUS_CODE"].fillna(-1) > 3).mean()).rename(f"w{w}_share_dpd_gt3")
        share_dpd_gt4 = grp.apply(lambda g: (g["STATUS_CODE"].fillna(-1) > 4).mean()).rename(f"w{w}_share_dpd_gt4")
        share_unknown = grp.apply(lambda g: (g["STATUS"] == "X").mean()).rename(f"w{w}_share_unknown")
        share_closed = grp.apply(lambda g: (g["STATUS"] == "C").mean()).rename(f"w{w}_share_closed")

        # months since last dpd>0 to -1
        def _msld(g: pd.DataFrame) -> float:
            g = g.sort_values("MONTHS_BALANCE")  # ascending toward -1
            delin = g.loc[g["STATUS_CODE"] > 0, "MONTHS_BALANCE"]
            if delin.empty:
                return np.nan
            last_delin = delin.max()
            return float(-1 - last_delin)  # distance from last delinquency month to -1
        months_since_last_dpd = grp.apply(_msld).rename(f"w{w}_months_since_last_dpd_gt0")

        # longest consecutive delinquent streak
        def _streak(g: pd.DataFrame) -> int:
            g = g.sort_values("MONTHS_BALANCE")
            flags = (g["STATUS_CODE"].fillna(-1) > 0).astype(int)
            return _run_length_encode(flags)
        longest_streak = grp.apply(_streak).rename(f"w{w}_longest_streak_dpd_gt0")

        # status mix (fractions for 0..5)
        mixes = []
        for code in [0,1,2,3,4,5]:
            col = f"w{w}_status_{code}_share"
            mixes.append(grp.apply(lambda g, c=code: (g["STATUS_CODE"] == c).mean()).rename(col))

        block = pd.concat([
            months_count, dpd_mean, dpd_max, dpd_sum,
            share_dpd_gt0, share_dpd_gt1, share_dpd_gt2, share_dpd_gt3, share_dpd_gt4, share_unknown, share_closed,
            months_since_last_dpd, longest_streak, *mixes
        ], axis=1).reset_index()
        outs.append(block)

    out = outs[0]
    for b in outs[1:]:
        out = out.merge(b, on="SK_BUREAU_ID", how="left")
    return out


def agg_bb_to_customer(bureau: pd.DataFrame, bb_feats: pd.DataFrame, *, aggs: dict | None = None) -> pd.DataFrame:
    if aggs is None:
        aggs = {"mean": ["mean"], "max": ["max"], "sum": ["sum"]}
    b2 = bureau[["SK_ID_CURR", "SK_BUREAU_ID"]].drop_duplicates()
    joined = b2.merge(bb_feats, on="SK_BUREAU_ID", how="left")
    feat_cols = [c for c in joined.columns if c not in ("SK_ID_CURR", "SK_BUREAU_ID")]
    # Build aggregation dict from requested ops
    agg_ops = {}
    for c in feat_cols:
        ops = []
        for k, v in aggs.items():
            ops.extend(v)
        agg_ops[c] = list(dict.fromkeys(ops))  # dedupe while preserving order
    g = joined.groupby("SK_ID_CURR").agg(agg_ops)
    g.columns = [f"bb_{c}__{stat}" for c, stat in g.columns]
    return g.reset_index()


def compute_bureau_balance_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
    *,
    months_windows: list[int] = [6, 12, 24],
) -> pd.DataFrame:
    """
    Convenience function: from raw tables to customer-level bb features.
    1) Compute per-account bureau_balance features
    2) Aggregate to SK_ID_CURR via bureau mapping
    """
    bbp = bb_features_per_account(bureau_balance, months_windows=months_windows)
    cust = agg_bb_to_customer(bureau, bbp)
    return cust


# if __name__ == "__main__":
#     # Tiny smoke test for the bb pipeline
#     bureau_ex = pd.DataFrame({
#         "SK_ID_CURR": [1,1,2],
#         "SK_BUREAU_ID": [10,11,20],
#     })
#     bb_ex = pd.DataFrame({
#         "SK_BUREAU_ID": [10,10,10,11,11,20,20],
#         "MONTHS_BALANCE": [-3,-2,-1,-2,-1,-1,-5],
#         "STATUS": ["0","1","0","C","C","2","X"],
#     })
#     bb_feats = compute_bureau_balance_features(bureau_ex, bb_ex, months_windows=[6,12])
#     print(bb_feats.head())

# ===================================================================================== #
# Application Table Feature Engineering
# ===================================================================================== #


# =========================== application features =========================== #
# Feature engineering for the application table (one row per SK_ID_CURR).

APP_COMM_FLAGS = [
    "FLAG_MOBIL","FLAG_EMP_PHONE","FLAG_WORK_PHONE","FLAG_CONT_MOBILE","FLAG_PHONE","FLAG_EMAIL"
]

APP_MISMATCH_FLAGS_REGION = [
    "REG_REGION_NOT_LIVE_REGION","REG_REGION_NOT_WORK_REGION","LIVE_REGION_NOT_WORK_REGION"
]

APP_MISMATCH_FLAGS_CITY = [
    "REG_CITY_NOT_LIVE_CITY","REG_CITY_NOT_WORK_CITY","LIVE_CITY_NOT_WORK_CITY"
]

EXT_SOURCES = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]


def _freq_encode(s: pd.Series) -> pd.Series:
    """Frequency-encode a categorical series (NaN-safe)."""
    vc = s.value_counts(dropna=False)
    return s.map(vc).astype(float)


def prep_application(app: pd.DataFrame) -> pd.DataFrame:
    """Basic normalization & core derived columns.
    - Convert DAYS_* to positive years where appropriate
    - Clean categorical text (lower/strip)
    - Safe division for ratios
    """
    d = app.copy()
    # numeric derived
    if "DAYS_BIRTH" in d:
        d["AGE_YEARS"] = (-d["DAYS_BIRTH"].astype(float)) / 365.0
    if "DAYS_EMPLOYED" in d:
        d["EMPLOY_YEARS"] = (-d["DAYS_EMPLOYED"].astype(float)) / 365.0
    if "DAYS_REGISTRATION" in d:
        d["REG_YEARS_AGO"] = (-d["DAYS_REGISTRATION"].astype(float)) / 365.0
    if "DAYS_ID_PUBLISH" in d:
        d["ID_PUBLISH_YEARS_AGO"] = (-d["DAYS_ID_PUBLISH"].astype(float)) / 365.0
    if "DAYS_LAST_PHONE_CHANGE" in d:
        d["PHONE_CHANGE_YEARS_AGO"] = (-d["DAYS_LAST_PHONE_CHANGE"].astype(float)) / 365.0

    # ratios (safe)
    if set(["AMT_CREDIT","AMT_INCOME_TOTAL"]).issubset(d):
        d["CREDIT_TO_INCOME"] = _safe_div(d["AMT_CREDIT"], d["AMT_INCOME_TOTAL"])
    if set(["AMT_ANNUITY","AMT_INCOME_TOTAL"]).issubset(d):
        d["ANNUITY_TO_INCOME"] = _safe_div(d["AMT_ANNUITY"], d["AMT_INCOME_TOTAL"])
    if set(["AMT_ANNUITY","AMT_CREDIT"]).issubset(d):
        d["ANNUITY_TO_CREDIT"] = _safe_div(d["AMT_ANNUITY"], d["AMT_CREDIT"])  # term proxy
    if set(["AMT_GOODS_PRICE","AMT_CREDIT"]).issubset(d):
        d["GOODS_TO_CREDIT"] = _safe_div(d["AMT_GOODS_PRICE"], d["AMT_CREDIT"]) 
    if set(["AMT_INCOME_TOTAL","CNT_FAM_MEMBERS"]).issubset(d):
        d["INCOME_PER_FAM"] = _safe_div(d["AMT_INCOME_TOTAL"], d["CNT_FAM_MEMBERS"]) 
    if set(["CNT_CHILDREN","CNT_FAM_MEMBERS"]).issubset(d):
        d["CHILDREN_PER_FAM"] = _safe_div(d["CNT_CHILDREN"], d["CNT_FAM_MEMBERS"]) 
    if set(["EMPLOY_YEARS","AGE_YEARS"]).issubset(d):
        d["TENURE_OVER_AGE"] = _safe_div(d["EMPLOY_YEARS"], d["AGE_YEARS"]) 

    # counts / sums
    comm_cols = [c for c in APP_COMM_FLAGS if c in d.columns]
    if comm_cols:
        d["COMM_CHANNELS_COUNT"] = d[comm_cols].sum(axis=1)
    reg_cols = [c for c in APP_MISMATCH_FLAGS_REGION if c in d.columns]
    city_cols = [c for c in APP_MISMATCH_FLAGS_CITY if c in d.columns]
    if reg_cols:
        d["ADDR_MISMATCH_REGION_SUM"] = d[reg_cols].sum(axis=1)
    if city_cols:
        d["ADDR_MISMATCH_CITY_SUM"] = d[city_cols].sum(axis=1)
    if reg_cols or city_cols:
        d["ADDR_MISMATCH_SUM"] = d[[*(reg_cols or []), *(city_cols or [])]].sum(axis=1)

    # external source aggregates
    xs = [c for c in EXT_SOURCES if c in d.columns]
    if xs:
        d["EXT_MIN"] = d[xs].min(axis=1)
        d["EXT_MAX"] = d[xs].max(axis=1)
        d["EXT_MEAN"] = d[xs].mean(axis=1)
        d["EXT_STD"] = d[xs].std(axis=1)
        d["EXT_VAR"] = d[xs].var(axis=1)
        # disagreement magnitude among sources
        d["EXT_SPAN"] = d["EXT_MAX"] - d["EXT_MIN"]

    # enquiries
    req_cols = [c for c in d.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if req_cols:
        d["REQ_TOTAL"] = d[req_cols].sum(axis=1)
        recent_cols = [c for c in req_cols if any(k in c for k in ["HOUR","DAY","WEEK","MON","QRT"])]
        long_cols = [c for c in req_cols if "YEAR" in c]
        if recent_cols:
            d["REQ_RECENT"] = d[recent_cols].sum(axis=1)
            d["REQ_RECENT_RATIO"] = _safe_div(d["REQ_RECENT"], d["REQ_TOTAL"])
        if long_cols:
            d["REQ_YEAR_ONLY"] = d[long_cols].sum(axis=1)

    # assets & docs
    if {"FLAG_OWN_CAR","FLAG_OWN_REALTY"}.issubset(d.columns):
        d["ASSET_FLAGS_SUM"] = d["FLAG_OWN_CAR"].astype(float) + d["FLAG_OWN_REALTY"].astype(float)
    doc_cols = [c for c in d.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        d["DOCS_COUNT"] = d[doc_cols].sum(axis=1)
        d["DOCS_MISSING_RATIO"] = (len(doc_cols) - d["DOCS_COUNT"].astype(float))/ float(len(doc_cols))

    # timing features
    if "HOUR_APPR_PROCESS_START" in d:
        h = d["HOUR_APPR_PROCESS_START"].astype(float)
        d["HOUR_SIN"] = np.sin(2 * np.pi * h / 24.0)
        d["HOUR_COS"] = np.cos(2 * np.pi * h / 24.0)
        d["HOUR_LATE_FLAG"] = (h >= 20) | (h <= 6)
        d["HOUR_LATE_FLAG"] = d["HOUR_LATE_FLAG"].astype(float)
    if "WEEKDAY_APPR_PROCESS_START" in d:
        # keep raw; optionally frequency-encode as numeric helper
        d["WEEKDAY_FREQ"] = _freq_encode(d["WEEKDAY_APPR_PROCESS_START"].astype(str))

    # environment aggregates (normalized building variables)
    env_cols = [c for c in d.columns if c.endswith(('_AVG','_MODE','_MEDI'))]
    if env_cols:
        #d["ENV_MEAN"] = d[env_cols].mean(axis=1)
        d["ENV_MISSING_CNT"] = d[env_cols].isna().sum(axis=1)
        d["ENV_COVERAGE"] = 1.0 - d["ENV_MISSING_CNT"].astype(float)/float(len(env_cols))

    # categorical cleaning for downstream encoding (kept as text)
    for cat in [
        "NAME_CONTRACT_TYPE","CODE_GENDER","NAME_TYPE_SUITE","NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE",
        "OCCUPATION_TYPE","ORGANIZATION_TYPE",
    ]:
        if cat in d.columns:
            d[cat] = d[cat].astype(str).str.strip().str.lower()
            d[f"{cat}_FREQ"] = _freq_encode(d[cat])

    return d


def compute_application_features(app: pd.DataFrame) -> pd.DataFrame:
    """Public entry point: returns engineered application-level features.
    Output columns are additive to input; function returns only [SK_ID_CURR + new feature cols].
    """
    d = prep_application(app)
    keep = ["SK_ID_CURR"] + [c for c in d.columns if c != "SK_ID_CURR" and c not in app.columns]
    return d[keep].copy()
