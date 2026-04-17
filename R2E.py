"""
Streamlit R2E dashboard — Retention tab (Looker tab 4 migration).

Data: BigQuery cohort metrics with cohort_month, month_post_ft, and either
      (num, denum) for Looker-style SUM(num)/SUM(denum), or legacy user_cnt
      (retention = users at M+k divided by cohort size at M+0).
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

try:
    from google.oauth2 import service_account
except ImportError:  # pragma: no cover
    service_account = None

try:
    from pandas_gbq import read_gbq
except ImportError:  # pragma: no cover
    read_gbq = None

try:
    from pandas_gbq.exceptions import LargeResultsWarning
except ImportError:  # pragma: no cover
    LargeResultsWarning = None  # type: ignore[misc, assignment]


def _bq_credentials_from_secrets() -> Any | None:
    """Optional service account JSON in Streamlit secrets as `gcp_service_account`."""
    if service_account is None:
        return None
    try:
        raw = st.secrets.get("gcp_service_account")
    except Exception:
        return None
    if raw is None:
        return None
    try:
        info = json.loads(raw) if isinstance(raw, str) else dict(raw)
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    except Exception:
        return None


def _bq_project_id(full_table_id: str) -> str:
    parts = full_table_id.split(".")
    if len(parts) != 3 or not all(parts):
        raise ValueError("Use fully qualified table id: project_id.dataset.table")
    return parts[0]


def _read_gbq_sql(full_table_id: str, query: str) -> pd.DataFrame:
    if read_gbq is None:
        raise RuntimeError(
            "pandas-gbq is not installed. Run: pip install pandas-gbq google-cloud-bigquery"
        )
    creds = _bq_credentials_from_secrets()
    kw: dict[str, Any] = dict(dialect="standard", use_bqstorage_api=False)
    if creds is not None:
        kw["credentials"] = creds
    with warnings.catch_warnings():
        if LargeResultsWarning is not None:
            warnings.simplefilter("ignore", LargeResultsWarning)
        warnings.filterwarnings(
            "ignore",
            message=".*progress bar.*tqdm.*",
            category=UserWarning,
        )
        return read_gbq(query, project_id=_bq_project_id(full_table_id), **kw)


def _sql_ident(col: str) -> str:
    """BigQuery escaped identifier (handles mixed-case columns)."""
    safe = col.replace("`", "")
    return f"`{safe}`"


def _sql_literal(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if bool(v) else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))
    if isinstance(v, (datetime, date, pd.Timestamp)):
        ts = pd.Timestamp(v)
        return f"DATE '{ts.date().isoformat()}'"
    s = str(v).replace("\\", "\\\\").replace("'", "''")
    return f"'{s}'"


def _sql_in_list(ident: str, values: list[Any]) -> str:
    if not values:
        return "TRUE"
    inner = ", ".join(_sql_literal(v) for v in values)
    return f"{ident} IN ({inner})"


def _dataframe_all_strings_for_display(
    df: pd.DataFrame, na_rep: str = "—"
) -> pd.DataFrame:
    """
    Streamlit serializes dataframes with PyArrow; object columns that mix numbers
    and placeholder strings can infer as int64 and fail. Force every cell to str.
    """
    return df.map(lambda v: na_rep if pd.isna(v) else str(v))


def _sql_active_rider_is_one(arf_column: str) -> str:
    """Matches INT 1 or STRING '1' (common warehouse variants)."""
    i = _sql_ident(arf_column)
    return f"({i} = 1 OR CAST({i} AS STRING) = '1')"


@dataclass
class SqlFilterState:
    """WHERE fragments and cohort range from the filter UI (server-side mode)."""

    where_parts: list[str]
    cohort_start: pd.Timestamp | None
    cohort_end: pd.Timestamp | None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# project.dataset.table — first segment is billing project for the BQ job.
DEFAULT_TABLE = "safetyp.New_R2E_dashboard.R2E_Retention_tab_metrics"
MAX_MONTH_POST = 12  # columns 0 .. 12 inclusive

# Dimension filters (Looker "Month" on cohort is cohort_month below)
FILTER_DIMENSIONS: list[tuple[str, ...]] = [
    ("mega_region",),
    ("country_name",),
    ("active_rider_flag",),
    ("rider_frequency",),
    ("is_member",),
    ("Active_rider_eater_segment", "Active_rider_eater_Segment"),
    ("fraud_flag",),
    ("platform",),
    ("eater_segment",),
    ("eater_super_segment",),
    ("eater_detailed_segment",),
]

COHORT_ALIASES = ("cohort_month", "Month", "month", "Cohort Month")


def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _resolve_schema(
    df: pd.DataFrame,
    filter_cols_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Map logical names -> actual column names in the dataframe."""
    out: dict[str, Any] = {}

    cm = _find_column(df, *COHORT_ALIASES)
    if not cm:
        raise ValueError(
            "Could not find cohort month column (expected cohort_month or Month)."
        )
    out["cohort_month"] = cm

    mp = _find_column(df, "month_post_ft", "month_post_FT")
    if not mp:
        raise ValueError("Could not find month_post_ft column.")
    out["month_post_ft"] = mp

    num_c = _find_column(df, "num", "numerator")
    den_c = _find_column(df, "denum", "denominator", "denom")
    uc = _find_column(df, "user_cnt", "users", "user_count")
    if num_c and den_c:
        out["metric_mode"] = "num_denum"
        out["num"] = num_c
        out["denum"] = den_c
    elif uc:
        out["metric_mode"] = "user_cnt"
        out["user_cnt"] = uc
    else:
        raise ValueError(
            "Expected either (num, denum) columns or user_cnt for retention."
        )

    if filter_cols_override is not None:
        out["filter_cols"] = dict(filter_cols_override)
        return out

    out["filter_cols"] = {}
    for names in FILTER_DIMENSIONS:
        key = names[0]
        found = None
        for cand in names:
            resolved = _find_column(df, cand)
            if resolved:
                found = resolved
                break
        if not found:
            raise ValueError(f"Missing dimension column for {key}.")
        out["filter_cols"][key] = found

    return out


def _normalize_types(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    x = df.copy()
    cm = schema["cohort_month"]
    x[cm] = pd.to_datetime(x[cm], errors="coerce")
    x = x.dropna(subset=[cm])
    x[cm] = x[cm].dt.to_period("M").dt.to_timestamp()

    mp = schema["month_post_ft"]
    x[mp] = pd.to_numeric(x[mp], errors="coerce")
    x = x.dropna(subset=[mp])
    x[mp] = x[mp].astype(int)

    if schema.get("metric_mode") == "num_denum":
        x[schema["num"]] = pd.to_numeric(x[schema["num"]], errors="coerce").fillna(0)
        x[schema["denum"]] = pd.to_numeric(x[schema["denum"]], errors="coerce").fillna(
            0
        )
    else:
        uc = schema["user_cnt"]
        x[uc] = pd.to_numeric(x[uc], errors="coerce").fillna(0)

    return x


@st.cache_data(ttl=3600, show_spinner="Reading table schema…")
def load_empty_schema_df(full_table_id: str) -> pd.DataFrame:
    """Zero-row read: column names/dtypes only (minimal BigQuery bytes)."""
    return _read_gbq_sql(full_table_id, f"SELECT * FROM `{full_table_id}` LIMIT 0")


@st.cache_data(ttl=1800, show_spinner="Loading cohort date bounds…")
def load_cohort_bounds_bq(
    full_table_id: str, cohort_column: str, active_rider_column: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    cm = _sql_ident(cohort_column)
    arf_one = _sql_active_rider_is_one(active_rider_column)
    q = (
        f"SELECT MIN({cm}) AS mn, MAX({cm}) AS mx "
        f"FROM `{full_table_id}` WHERE {arf_one}"
    )
    row = _read_gbq_sql(full_table_id, q).iloc[0]
    mn = pd.to_datetime(row["mn"], utc=False, errors="coerce")
    mx = pd.to_datetime(row["mx"], utc=False, errors="coerce")
    if pd.isna(mn) or pd.isna(mx):
        raise ValueError("Could not read cohort month min/max (empty table?).")
    return mn, mx


def _distinct_values_query(full_table_id: str, column: str, arf_column: str) -> str:
    """Distinct values for a dimension, scoped to active riders (faster when clustered)."""
    c = _sql_ident(column)
    arf_one = _sql_active_rider_is_one(arf_column)
    return (
        f"SELECT DISTINCT {c} AS v FROM `{full_table_id}` "
        f"WHERE {arf_one} AND {c} IS NOT NULL"
    )


def _distinct_active_rider_flags_query(full_table_id: str, column: str) -> str:
    """Do not filter on active_rider_flag here, or DISTINCT would only ever return 1."""
    c = _sql_ident(column)
    return f"SELECT DISTINCT {c} AS v FROM `{full_table_id}` WHERE {c} IS NOT NULL"


def _merge_ar_flag_options(vals: list[Any]) -> list[Any]:
    """Ensure 0 and 1 appear in the multiselect when absent from the table sample."""
    out = list(vals)
    for bit in (0, 1):
        if not any(v == bit or str(v) == str(bit) for v in out):
            out.append(bit)
    return sorted(out, key=lambda x: (str(type(x).__name__), str(x)))


@st.cache_data(ttl=1800, show_spinner="Loading filter dropdown values from BigQuery…")
def load_filter_option_lists_bq(
    full_table_id: str,
    filter_cols_json: str,
    active_rider_sql_column: str,
) -> dict[str, list[Any]]:
    """
    One DISTINCT query per dimension (parallel). Most dims are scoped with
    active_rider_flag = 1 for smaller scans; active_rider_flag itself is queried
    without that predicate so both 0 and 1 appear in the dropdown.
    """
    mapping: dict[str, str] = json.loads(filter_cols_json)
    out: dict[str, list[Any]] = {}

    def _run_one(logical: str, col: str) -> tuple[str, list[Any]]:
        if logical == "active_rider_flag":
            q = _distinct_active_rider_flags_query(full_table_id, col)
        else:
            q = _distinct_values_query(full_table_id, col, active_rider_sql_column)
        df = _read_gbq_sql(full_table_id, q)
        vals = sorted(df["v"].dropna().unique().tolist(), key=lambda x: str(x))
        if logical == "active_rider_flag":
            vals = _merge_ar_flag_options(vals)
        return logical, vals

    with ThreadPoolExecutor(max_workers=min(10, max(1, len(mapping)))) as pool:
        futures = {
            pool.submit(_run_one, logical, col): logical for logical, col in mapping.items()
        }
        for fut in as_completed(futures):
            logical, vals = fut.result()
            out[logical] = vals
    return out


@st.cache_data(ttl=600, show_spinner="Running aggregated retention query…")
def load_retention_aggregate_bq(
    full_table_id: str,
    metric_mode: str,
    cohort_col: str,
    month_post_col: str,
    num_col: str,
    den_col: str,
    user_cnt_col: str,
    where_sql: str,
) -> pd.DataFrame:
    """
    Server-side roll-up: only cohort_month × month_post_ft leaves BigQuery.
    Equivalent to pandas groupby + SUM after applying the same WHERE filters.
    """
    tref = f"`{full_table_id}`"
    cm, mp = _sql_ident(cohort_col), _sql_ident(month_post_col)
    w = where_sql.strip() if where_sql.strip() else "TRUE"
    if metric_mode == "num_denum":
        n, d = _sql_ident(num_col), _sql_ident(den_col)
        q = (
            f"SELECT {cm}, {mp}, "
            f"SUM({n}) AS num, SUM({d}) AS denum "
            f"FROM {tref} WHERE {w} GROUP BY 1, 2"
        )
    else:
        uc = _sql_ident(user_cnt_col)
        q = (
            f"SELECT {cm}, {mp}, "
            f"SUM({uc}) AS user_cnt "
            f"FROM {tref} WHERE {w} GROUP BY 1, 2"
        )
    return _read_gbq_sql(full_table_id, q)


@st.cache_data(ttl=900, show_spinner="Loading full table (may be slow)…")
def load_retention_full_table_bq(full_table_id: str) -> pd.DataFrame:
    """Legacy path: every row (use only for small tables or debugging)."""
    return _read_gbq_sql(full_table_id, f"SELECT * FROM `{full_table_id}`")


def _aggregate_for_pivot(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    cm = schema["cohort_month"]
    mp = schema["month_post_ft"]
    if schema.get("metric_mode") == "num_denum":
        n, d = schema["num"], schema["denum"]
        return df.groupby([cm, mp], as_index=False).agg(
            num_sum=(n, "sum"),
            den_sum=(d, "sum"),
        )
    uc = schema["user_cnt"]
    return df.groupby([cm, mp], as_index=False)[uc].sum().rename(columns={uc: "uc_sum"})


def build_retention_pivot(
    df: pd.DataFrame,
    schema: dict[str, Any],
    as_of: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (retention_pct, auxiliary_pivot) with cohort_month as rows and
    month_post_ft 0..MAX_MONTH_POST as columns.

    num_denum mode (Looker): each cell is SUM(num)/SUM(denum) for that cohort and offset.
    user_cnt mode (legacy): retention = sum(users at offset m) / sum(users at m=0).
    """
    cm = schema["cohort_month"]
    mp = schema["month_post_ft"]

    agg = _aggregate_for_pivot(df, schema)

    if schema.get("metric_mode") == "num_denum":
        den_sum = agg["den_sum"].replace(0, np.nan)
        agg = agg.assign(rate=np.where(den_sum.notna(), agg["num_sum"] / den_sum, np.nan))
        retention = agg.pivot(index=cm, columns=mp, values="rate")
        # Second grid: summed denum (useful for QA / weighting)
        counts = agg.pivot(index=cm, columns=mp, values="den_sum")
    else:
        retention = agg.pivot(index=cm, columns=mp, values="uc_sum")
        counts = retention.copy()
        retention = retention.reindex(columns=range(MAX_MONTH_POST + 1))
        counts = counts.reindex(columns=range(MAX_MONTH_POST + 1))
        if 0 not in retention.columns or retention[0].isna().all():
            raise ValueError("No month_post_ft = 0 (cohort size) in filtered data.")
        denom = retention[0].replace(0, np.nan)
        retention = retention.div(denom, axis=0)

    retention = retention.reindex(columns=range(MAX_MONTH_POST + 1))
    counts = counts.reindex(columns=range(MAX_MONTH_POST + 1))

    # Staircase: future months not yet observable -> NaN (displayed as —)
    if as_of is None:
        as_of = date.today()
    as_of_ts = pd.Timestamp(as_of).to_period("M").to_timestamp()
    for cohort in retention.index:
        cm_ts = pd.Timestamp(cohort)
        for m in range(MAX_MONTH_POST + 1):
            horizon = cm_ts + pd.DateOffset(months=m)
            if horizon > as_of_ts and m > 0:
                retention.loc[cohort, m] = np.nan
                counts.loc[cohort, m] = np.nan

    counts.columns = [str(c) for c in counts.columns]
    retention.columns = [str(c) for c in retention.columns]

    def _format_cohort_labels(idx: pd.Index) -> pd.Index:
        labels = []
        for x in idx:
            ts = pd.Timestamp(x)
            labels.append(f"{ts.strftime('%b')} {ts.day}, {ts.year}")
        return pd.Index(labels)

    counts.index = _format_cohort_labels(counts.index)
    retention.index = counts.index

    return retention, counts


def _style_retention_heatmap(retention: pd.DataFrame) -> Styler:
    """Green heatmap similar to Looker; empty cells stay light."""

    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return "—"
        return f"{float(v):.1%}"

    numeric = retention.apply(pd.to_numeric, errors="coerce")
    return retention.style.format(_fmt, na_rep="—").background_gradient(
        cmap="Greens",
        axis=None,
        vmin=0.0,
        vmax=1.0,
        gmap=numeric,
    )


def dimension_filters(
    df: pd.DataFrame,
    schema: dict[str, Any],
) -> pd.DataFrame:
    """Apply Looker-style filters; default active_rider_flag = 1."""
    filtered = df.copy()
    fmap = schema["filter_cols"]

    st.markdown("**Filters**")

    # Row 1: five filters
    r1 = st.columns(5)
    # Row 2: five filters
    r2 = st.columns(5)
    # Row 3: last dimension + cohort month range (Looker "Month" = cohort_month)
    r3 = st.columns([4, 6])

    placements = [
        (r1[0], "mega_region"),
        (r1[1], "country_name"),
        (r1[2], "active_rider_flag"),
        (r1[3], "rider_frequency"),
        (r1[4], "is_member"),
        (r2[0], "Active_rider_eater_segment"),
        (r2[1], "fraud_flag"),
        (r2[2], "platform"),
        (r2[3], "eater_segment"),
        (r2[4], "eater_super_segment"),
        (r3[0], "eater_detailed_segment"),
    ]

    cm_col = schema["cohort_month"]

    for col_slot, logical in placements:
        actual = fmap[logical]
        with col_slot:
            opts = sorted(filtered[actual].dropna().unique().tolist(), key=lambda x: str(x))
            if logical == "active_rider_flag":
                opts = _merge_ar_flag_options(opts)
                default = [x for x in opts if str(x) == "1" or x == 1]
                if not default and opts:
                    default = [opts[0]]
                sel = st.multiselect(
                    logical.replace("_", " ").title(),
                    options=opts,
                    default=default,
                    key=f"flt_{logical}",
                )
            else:
                sel = st.multiselect(
                    logical.replace("_", " ").title(),
                    options=opts,
                    default=[],
                    key=f"flt_{logical}",
                )
            if sel:
                filtered = filtered[filtered[actual].isin(sel)]

    cm_series = pd.to_datetime(filtered[cm_col], errors="coerce")
    min_m = cm_series.min()
    max_m = cm_series.max()
    with r3[1]:
        st.caption("Filters cohort rows (first order month).")
        if pd.notna(min_m) and pd.notna(max_m):
            dr = st.date_input(
                "Cohort month range",
                value=(min_m.date(), max_m.date()),
                min_value=min_m.date(),
                max_value=max_m.date(),
                key="cohort_month_range",
            )
            if isinstance(dr, tuple) and len(dr) == 2:
                a, b = dr
                ts_a = pd.Timestamp(a)
                ts_b = pd.Timestamp(b)
                filtered = filtered[(cm_series >= ts_a) & (cm_series <= ts_b)]

    return filtered


def server_side_filter_widgets(
    schema: dict[str, Any],
    option_lists: dict[str, list[Any]],
    cohort_min: pd.Timestamp,
    cohort_max: pd.Timestamp,
) -> SqlFilterState:
    """Multiselects backed by precomputed DISTINCT lists; emits SQL WHERE fragments."""
    where_parts: list[str] = []
    fmap = schema["filter_cols"]

    st.markdown("**Filters**")

    r1 = st.columns(5)
    r2 = st.columns(5)
    r3 = st.columns([4, 6])

    placements = [
        (r1[0], "mega_region"),
        (r1[1], "country_name"),
        (r1[2], "active_rider_flag"),
        (r1[3], "rider_frequency"),
        (r1[4], "is_member"),
        (r2[0], "Active_rider_eater_segment"),
        (r2[1], "fraud_flag"),
        (r2[2], "platform"),
        (r2[3], "eater_segment"),
        (r2[4], "eater_super_segment"),
        (r3[0], "eater_detailed_segment"),
    ]

    cohort_sql = _sql_ident(schema["cohort_month"])

    for col_slot, logical in placements:
        actual = fmap[logical]
        with col_slot:
            opts = list(option_lists.get(logical, []))
            if logical == "active_rider_flag":
                opts = _merge_ar_flag_options(opts)
                default = [x for x in opts if str(x) == "1" or x == 1]
                if not default and opts:
                    default = [opts[0]]
                sel = st.multiselect(
                    logical.replace("_", " ").title(),
                    options=opts,
                    default=default,
                    key=f"flt_{logical}",
                )
            else:
                sel = st.multiselect(
                    logical.replace("_", " ").title(),
                    options=opts,
                    default=[],
                    key=f"flt_{logical}",
                )
            if sel:
                where_parts.append(_sql_in_list(_sql_ident(actual), sel))

    cohort_start: pd.Timestamp | None = None
    cohort_end: pd.Timestamp | None = None
    with r3[1]:
        st.caption("Filters cohort rows (first order month) in SQL.")
        if pd.notna(cohort_min) and pd.notna(cohort_max):
            dr = st.date_input(
                "Cohort month range",
                value=(cohort_min.date(), cohort_max.date()),
                min_value=cohort_min.date(),
                max_value=cohort_max.date(),
                key="cohort_month_range",
            )
            if isinstance(dr, tuple) and len(dr) == 2:
                a, b = dr
                cohort_start = pd.Timestamp(a)
                cohort_end = pd.Timestamp(b)
                where_parts.append(
                    f"DATE({cohort_sql}) BETWEEN DATE('{a.isoformat()}') "
                    f"AND DATE('{b.isoformat()}')"
                )

    return SqlFilterState(where_parts=where_parts, cohort_start=cohort_start, cohort_end=cohort_end)


def main() -> None:
    st.set_page_config(page_title="R2E Dashboard", layout="wide")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; max-width: 100%; }
        div[data-testid="stVerticalBlock"] > div:has(> label) { margin-bottom: 0.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar is global; keep it outside tabs so it always runs and matches the UI.
    with st.sidebar:
        st.header("Data source")
        try:
            def_table = st.secrets.get("BQ_FULL_TABLE", DEFAULT_TABLE)
        except Exception:
            def_table = DEFAULT_TABLE

        full_table = st.text_input(
            "Fully qualified table",
            value=def_table,
            help="Format: project_id.dataset.table. The job is billed to project_id (first segment).",
        )
        use_sql_aggregate = st.checkbox(
            "BigQuery GROUP BY for the heatmap (faster download)",
            value=False,
            help=(
                "Unchecked (default): SELECT * — every row is loaded into Python before filters "
                "and pivot. Use this when you require the full granular table in memory.\n\n"
                "Checked: only the rolled-up cohort × month_post_ft grid is downloaded. For "
                "additive metrics SUM(num) and SUM(denum), that is the same result as summing "
                "all filtered rows in pandas (no change to the percentages); you only skip "
                "storing millions of unused detail rows in this process."
            ),
        )
        if st.button("Refresh data cache"):
            st.cache_data.clear()

    # Retention first so http://localhost:8501 opens on the real dashboard (not a placeholder).
    tab_retention, tab1, tab2, tab3, tab5 = st.tabs(
        ["Retention", "Tab 1", "Tab 2", "Tab 3", "Tab 5"]
    )

    with tab1:
        st.info("Placeholder — migrate when ready.")
    with tab2:
        st.info("Placeholder — migrate when ready.")
    with tab3:
        st.info("Placeholder — migrate when ready.")

    with tab5:
        st.info("Placeholder — migrate when ready.")

    with tab_retention:
        st.subheader("Retention")
        st.caption(
            "Cohort retention by first-order month; columns are month_post_ft 0–12. "
            "Metric matches Looker: SUM(num) / SUM(denum) per cohort and offset."
        )

        if not full_table:
            st.warning("Set the fully qualified table id in the sidebar.")
            st.stop()

        try:
            if use_sql_aggregate:
                st.caption(
                    "Heatmap data is aggregated in BigQuery (GROUP BY + SUM). "
                    "Those sums match summing every filtered row in Python for this metric."
                )
                empty = load_empty_schema_df(full_table)
                schema_base = _resolve_schema(empty)
                arf_col = schema_base["filter_cols"]["active_rider_flag"]
                cohort_min, cohort_max = load_cohort_bounds_bq(
                    full_table, schema_base["cohort_month"], arf_col
                )
                filter_cols_json = json.dumps(schema_base["filter_cols"], sort_keys=True)
                option_lists = load_filter_option_lists_bq(
                    full_table, filter_cols_json, arf_col
                )
                sql_state = server_side_filter_widgets(
                    schema_base, option_lists, cohort_min, cohort_max
                )
                where_sql = (
                    " AND ".join(sql_state.where_parts)
                    if sql_state.where_parts
                    else "TRUE"
                )
                raw = load_retention_aggregate_bq(
                    full_table,
                    schema_base["metric_mode"],
                    schema_base["cohort_month"],
                    schema_base["month_post_ft"],
                    schema_base.get("num", ""),
                    schema_base.get("denum", ""),
                    schema_base.get("user_cnt", ""),
                    where_sql,
                )
                schema = _resolve_schema(
                    raw, filter_cols_override=schema_base["filter_cols"]
                )
            else:
                st.caption(
                    "Full table load: all rows are fetched, then filters and pivot run in Python."
                )
                raw = load_retention_full_table_bq(full_table)
                schema = _resolve_schema(raw)
        except Exception as exc:
            st.error(f"Could not load BigQuery table: {exc}")
            st.stop()

        df = _normalize_types(raw, schema)

        if use_sql_aggregate:
            filtered = df
            if filtered.empty:
                st.warning("No rows for the current SQL filters.")
                st.stop()
        else:
            filtered = dimension_filters(df, schema)
            if filtered.empty:
                st.warning("No rows after filters.")
                st.stop()

        try:
            retention, counts = build_retention_pivot(filtered, schema)
        except Exception as exc:
            st.error(f"Could not build pivot: {exc}")
            st.stop()

        st.markdown("##### Cohort retention rate")
        _df_width: dict[str, Any] = {"width": "stretch"}
        try:
            styled = _style_retention_heatmap(retention)
            st.dataframe(
                styled,
                **_df_width,
                height=min(900, 120 + 28 * len(retention)),
            )
        except Exception:
            st.dataframe(
                retention.map(lambda v: "—" if pd.isna(v) else f"{v:.1%}"),
                **_df_width,
            )

        aux_title = (
            "SUM(denum) by cohort × offset (QA)"
            if schema.get("metric_mode") == "num_denum"
            else "Cohort user counts (same pivot)"
        )
        with st.expander(aux_title):
            st.dataframe(_dataframe_all_strings_for_display(counts), **_df_width)

        csv_bytes = retention.rename_axis("cohort_month").reset_index().to_csv(
            index=False
        ).encode("utf-8")
        st.download_button(
            "Download retention CSV",
            data=csv_bytes,
            file_name="r2e_retention_pivot.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()