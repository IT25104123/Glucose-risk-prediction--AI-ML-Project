

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CODE DICTIONARY  (from the dataset's own Documentation file)
# ---------------------------------------------------------------------------
CODE_MAP = {
    33: ("Regular insulin dose",                 "insulin",  "regular"),
    34: ("NPH insulin dose",                     "insulin",  "nph"),
    35: ("UltraLente insulin dose",              "insulin",  "ultralente"),
    48: ("Unspecified blood glucose measurement","glucose",  "unspecified"),
    57: ("Unspecified blood glucose measurement","glucose",  "unspecified"),
    58: ("Pre-breakfast blood glucose",          "glucose",  "pre_breakfast"),
    59: ("Post-breakfast blood glucose",         "glucose",  "post_breakfast"),
    60: ("Pre-lunch blood glucose",              "glucose",  "pre_lunch"),
    61: ("Post-lunch blood glucose",             "glucose",  "post_lunch"),
    62: ("Pre-supper blood glucose",             "glucose",  "pre_supper"),
    63: ("Post-supper blood glucose",            "glucose",  "post_supper"),
    64: ("Pre-snack blood glucose",              "glucose",  "pre_snack"),
    65: ("Hypoglycemic symptoms",                "event",    "hypo_symptoms"),
    66: ("Typical meal ingestion",               "meal",     "typical"),
    67: ("More-than-usual meal ingestion",       "meal",     "more"),
    68: ("Less-than-usual meal ingestion",       "meal",     "less"),
    69: ("Typical exercise activity",            "exercise", "typical"),
    70: ("More-than-usual exercise activity",    "exercise", "more"),
    71: ("Less-than-usual exercise activity",    "exercise", "less"),
    72: ("Unspecified special event",            "event",    "special"),
}

# Code 56 is NOT in the dataset's own Data-Codes file, but appears 119 times
# across 24 patient files. Its values (min 31, median 130, max 501) sit squarely
# in the blood-glucose range and nowhere near insulin dosing (single digits),
# so it is almost certainly an undocumented glucose measurement.
# Off by default: an undocumented code is a judgement call you should make
# and be able to defend, not one the script makes silently for you.
UNDOCUMENTED_GLUCOSE_CODE = 56

# Clinical thresholds in mg/dL. Change here if your group agreed on others.
HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0

# Plausible physiological range; anything outside is a recording error.
GLUCOSE_MIN, GLUCOSE_MAX = 20.0, 500.0
INSULIN_MIN, INSULIN_MAX = 0.0, 100.0

# How far back to look when summing insulin / meals / exercise before a reading.
LOOKBACK_HOURS = 6


# ---------------------------------------------------------------------------
# 2. PARSING ONE RAW FILE
# ---------------------------------------------------------------------------
def clean_value(raw):
    """Raw values contain junk like '0Hi', "'008", '3.5', ''. Pull out a number."""
    if raw is None:
        return np.nan
    s = str(raw).strip().strip("'\"")
    if s == "" or s.lower() in {"na", "nan", "-"}:
        return np.nan
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan


def clean_time(raw):
    """Times appear as '8:00', '08:00', '0:00', sometimes malformed."""
    s = str(raw).strip()
    m = re.match(r"^(\d{1,2}):(\d{1,2})$", s)
    if not m:
        return np.nan
    h, mi = int(m.group(1)), int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mi <= 59):
        return np.nan
    return f"{h:02d}:{mi:02d}"


def parse_patient_file(path):
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            parts = re.split(r"\t+|\s{2,}", line.rstrip("\n"))
            parts = [p.strip() for p in parts if p.strip() != ""]
            if len(parts) < 3:
                rows.append({"_raw": line.rstrip(), "_line": lineno,
                             "date_raw": np.nan, "time_raw": np.nan,
                             "code": np.nan, "value": np.nan})
                continue
            date_raw, time_raw, code_raw = parts[0], parts[1], parts[2]
            value_raw = parts[3] if len(parts) > 3 else ""
            rows.append({
                "_raw": line.rstrip(),
                "_line": lineno,
                "date_raw": date_raw,
                "time_raw": time_raw,
                "code": clean_value(code_raw),
                "value": clean_value(value_raw),
            })
    df = pd.DataFrame(rows)
    df["patient_id"] = int(re.search(r"(\d+)", Path(path).name).group(1))
    return df


# ---------------------------------------------------------------------------
# 3. BUILD THE LONG (TIDY) TABLE
# ---------------------------------------------------------------------------
def build_long_table(input_dir, report, include_code_56=False):
    files = sorted(
        [p for p in Path(input_dir).iterdir()
         if p.is_file() and re.match(r"^data-\d+$", p.name)]
    )
    if not files:
        sys.exit(f"No data-NN files found in {input_dir}")

    report.append(f"Patient files found: {len(files)}")
    df = pd.concat([parse_patient_file(p) for p in files], ignore_index=True)
    report.append(f"Raw lines read:      {len(df)}")

    # --- date & time ---
    df["time"] = df["time_raw"].map(clean_time)
    df["date"] = pd.to_datetime(df["date_raw"], format="%m-%d-%Y", errors="coerce")

    bad_date = df["date"].isna().sum()
    bad_time = df["time"].isna().sum()
    report.append(f"Unparseable dates:   {bad_date}")
    report.append(f"Unparseable times:   {bad_time}")

    df = df.dropna(subset=["date", "time", "code"]).copy()
    df["code"] = df["code"].astype(int)

    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"]
    )

    # --- decode the code column ---
    code_map = dict(CODE_MAP)
    if include_code_56:
        code_map[UNDOCUMENTED_GLUCOSE_CODE] = (
            "Blood glucose (undocumented code 56)", "glucose", "undocumented_56")
        report.append("Code 56 INCLUDED as a glucose measurement (--include-code-56).")
    else:
        report.append("Code 56 EXCLUDED (undocumented; pass --include-code-56 to keep it).")

    fallback = ("Unknown code", "unknown", "unknown")
    df["code_description"] = df["code"].map(lambda c: code_map.get(c, fallback)[0])
    df["measurement_type"] = df["code"].map(lambda c: code_map.get(c, fallback)[1])
    df["context"]          = df["code"].map(lambda c: code_map.get(c, fallback)[2])

    unk = df.loc[df["measurement_type"] == "unknown", "code"]
    report.append(f"Rows with codes outside the documented list: {len(unk)}")
    if len(unk):
        detail = ", ".join(f"code {k} x{v}" for k, v in unk.value_counts().sort_index().items())
        report.append(f"  breakdown: {detail}")

    # --- range checks ---
    g = df["measurement_type"] == "glucose"
    i = df["measurement_type"] == "insulin"
    bad_g = g & (~df["value"].between(GLUCOSE_MIN, GLUCOSE_MAX))
    bad_i = i & (~df["value"].between(INSULIN_MIN, INSULIN_MAX))
    report.append(f"Glucose values outside {GLUCOSE_MIN:.0f}-{GLUCOSE_MAX:.0f} mg/dL set to NaN: {bad_g.sum()}")
    report.append(f"Insulin doses outside {INSULIN_MIN:.0f}-{INSULIN_MAX:.0f} units set to NaN:  {bad_i.sum()}")
    df.loc[bad_g | bad_i, "value"] = np.nan

    # --- duplicates ---
    before = len(df)
    df = df.drop_duplicates(subset=["patient_id", "timestamp", "code", "value"])
    report.append(f"Exact duplicate events removed: {before - len(df)}")

    df = df.sort_values(["patient_id", "timestamp", "code"]).reset_index(drop=True)

    cols = ["patient_id", "timestamp", "date", "time", "code",
            "code_description", "measurement_type", "context", "value"]
    long_df = df[cols].copy()
    long_df["date"] = long_df["date"].dt.date
    report.append(f"Final tidy event rows: {len(long_df)}")
    return long_df


# ---------------------------------------------------------------------------
# 4. BUILD THE MODELLING TABLE (one row per glucose reading)
# ---------------------------------------------------------------------------
def window_sum(events, patient, t0, t1, mask_col, mask_val, value=True):
    sub = events[(events["patient_id"] == patient) &
                 (events["timestamp"] > t0) &
                 (events["timestamp"] <= t1) &
                 (events[mask_col] == mask_val)]
    return sub["value"].sum() if value else len(sub)


def build_glucose_table(long_df, report):
    df = long_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    glucose = df[(df["measurement_type"] == "glucose") & df["value"].notna()].copy()
    report.append(f"Blood-glucose readings retained: {len(glucose)}")

    # ---- calendar features ----
    glucose["hour"] = glucose["timestamp"].dt.hour
    glucose["day_of_week"] = glucose["timestamp"].dt.day_name()
    glucose["is_weekend"] = glucose["timestamp"].dt.dayofweek >= 5
    glucose["time_of_day"] = pd.cut(
        glucose["hour"], bins=[-1, 5, 11, 17, 21, 24],
        labels=["night", "morning", "afternoon", "evening", "night_late"]
    ).astype(str)

    glucose = glucose.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    glucose = glucose.rename(columns={"value": "glucose_mg_dl"})

    # ---- lag / lead features per patient ----
    grp = glucose.groupby("patient_id")
    glucose["prev_glucose"] = grp["glucose_mg_dl"].shift(1)
    glucose["prev_glucose_2"] = grp["glucose_mg_dl"].shift(2)
    glucose["next_glucose"] = grp["glucose_mg_dl"].shift(-1)        # regression target
    glucose["hours_since_prev"] = grp["timestamp"].diff().dt.total_seconds() / 3600
    glucose["hours_to_next"] = -grp["timestamp"].diff(-1).dt.total_seconds() / 3600
    glucose["glucose_change"] = glucose["glucose_mg_dl"] - glucose["prev_glucose"]
    glucose["rolling_mean_3"] = grp["glucose_mg_dl"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    glucose["rolling_std_3"] = grp["glucose_mg_dl"].transform(
        lambda s: s.rolling(3, min_periods=2).std())
    glucose["reading_index"] = grp.cumcount() + 1

    # ---- insulin / meal / exercise in the preceding window ----
    events = df[df["measurement_type"].isin(["insulin", "meal", "exercise", "event"])].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    ins_cols = {"regular": "insulin_regular_6h",
                "nph": "insulin_nph_6h",
                "ultralente": "insulin_ultralente_6h"}
    for ctx, col in ins_cols.items():
        glucose[col] = 0.0
    glucose["meals_6h"] = 0
    glucose["exercise_6h"] = 0
    glucose["hypo_symptoms_6h"] = 0

    for pid, gsub in glucose.groupby("patient_id"):
        esub = events[events["patient_id"] == pid]
        if esub.empty:
            continue
        et = esub["timestamp"].values
        for idx, row in gsub.iterrows():
            t1 = np.datetime64(row["timestamp"])
            t0 = t1 - np.timedelta64(LOOKBACK_HOURS, "h")
            win = esub[(et > t0) & (et <= t1)]
            if win.empty:
                continue
            for ctx, col in ins_cols.items():
                v = win.loc[(win["measurement_type"] == "insulin") &
                            (win["context"] == ctx), "value"].sum()
                glucose.at[idx, col] = float(v)
            glucose.at[idx, "meals_6h"] = int((win["measurement_type"] == "meal").sum())
            glucose.at[idx, "exercise_6h"] = int((win["measurement_type"] == "exercise").sum())
            glucose.at[idx, "hypo_symptoms_6h"] = int((win["context"] == "hypo_symptoms").sum())

    glucose["total_insulin_6h"] = glucose[list(ins_cols.values())].sum(axis=1)

    # ---- daily insulin totals ----
    ins = df[df["measurement_type"] == "insulin"].copy()
    daily = (ins.groupby(["patient_id", "date"])["value"].sum()
                .rename("daily_total_insulin").reset_index())
    glucose = glucose.merge(daily, on=["patient_id", "date"], how="left")
    glucose["daily_total_insulin"] = glucose["daily_total_insulin"].fillna(0.0)

    # ---- targets ----
    def risk(v):
        if pd.isna(v):
            return np.nan
        if v < HYPO_THRESHOLD:
            return "Hypoglycemic"
        if v > HYPER_THRESHOLD:
            return "Hyperglycemic"
        return "Normal"

    glucose["risk_class"] = glucose["glucose_mg_dl"].map(risk)          # classification target
    glucose["next_risk_class"] = glucose["next_glucose"].map(risk)      # forward-looking target

    ordered = [
        "patient_id", "timestamp", "date", "time", "hour", "time_of_day",
        "day_of_week", "is_weekend", "reading_index",
        "code", "context", "code_description",
        "glucose_mg_dl",
        "prev_glucose", "prev_glucose_2", "glucose_change",
        "hours_since_prev", "hours_to_next",
        "rolling_mean_3", "rolling_std_3",
        "insulin_regular_6h", "insulin_nph_6h", "insulin_ultralente_6h",
        "total_insulin_6h", "daily_total_insulin",
        "meals_6h", "exercise_6h", "hypo_symptoms_6h",
        "risk_class", "next_glucose", "next_risk_class",
    ]
    glucose = glucose[ordered]

    report.append("Risk class distribution: " +
                  ", ".join(f"{k}={v}" for k, v in
                            glucose['risk_class'].value_counts().items()))
    return glucose


# ---------------------------------------------------------------------------
# 5. EXCEL EXPORT
# ---------------------------------------------------------------------------
def write_excel(path, long_df, glucose_df):
    """
    Write a .xlsx where date/time columns are REAL Excel date-times with an
    explicit display format and a wide enough column. Opening the CSV directly
    leaves Excel to guess, which is where '#####' and blank-looking cells
    come from.
    """
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter

    DATETIME_FMT = "yyyy-mm-dd hh:mm"
    DATE_FMT = "yyyy-mm-dd"
    TIME_FMT = "hh:mm"

    def prep(df):
        d = df.copy()
        if "timestamp" in d:
            d["timestamp"] = pd.to_datetime(d["timestamp"])
        if "date" in d:
            d["date"] = pd.to_datetime(d["date"])
        if "time" in d:
            # Excel stores a time-of-day as a fraction of a day (0.5 = 12:00).
            # Writing the text '08:00' leaves it as a left-aligned string that
            # the hh:mm format cannot act on.
            t = pd.to_datetime(d["time"], format="%H:%M")
            d["time"] = (t.dt.hour * 60 + t.dt.minute) / 1440.0
        return d

    sheets = {"glucose_table": prep(glucose_df), "event_log": prep(long_df)}

    with pd.ExcelWriter(path, engine="openpyxl",
                        datetime_format=DATETIME_FMT,
                        date_format=DATE_FMT) as xl:
        for name, d in sheets.items():
            d.to_excel(xl, sheet_name=name, index=False)
            ws = xl.book[name]
            ws.freeze_panes = "A2"

            for j, col in enumerate(d.columns, start=1):
                letter = get_column_letter(j)

                header = ws.cell(row=1, column=j)
                header.font = Font(name="Arial", bold=True)
                header.alignment = Alignment(horizontal="center")

                if col == "timestamp":
                    fmt, width = DATETIME_FMT, 20
                elif col == "date":
                    fmt, width = DATE_FMT, 13
                elif col == "time":
                    fmt, width = TIME_FMT, 10
                else:
                    fmt = None
                    width = max(len(str(col)) + 3, 12)

                if fmt:
                    for i in range(2, len(d) + 2):
                        ws.cell(row=i, column=j).number_format = fmt
                ws.column_dimensions[letter].width = width


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="diabetes-data",
                    help="folder containing data-01 ... data-70")
    ap.add_argument("--output", default="data", help="where to write the CSVs")
    ap.add_argument("--no-excel", action="store_true",
                    help="skip the .xlsx export")
    ap.add_argument("--include-code-56", action="store_true",
                    help="treat the undocumented code 56 as a glucose measurement")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = ["DATA QUALITY REPORT", "=" * 60]

    long_df = build_long_table(args.input, report, args.include_code_56)
    # utf-8-sig writes a BOM, which tells Excel the encoding and stops it
    # guessing the delimiter from your regional settings.
    long_df.to_csv(out / "diabetes_long.csv", index=False, encoding="utf-8-sig")

    report.append("-" * 60)
    glucose_df = build_glucose_table(long_df, report)
    glucose_df.to_csv(out / "diabetes_glucose.csv", index=False, encoding="utf-8-sig")

    report.append("-" * 60)
    report.append(f"Wrote {out/'diabetes_long.csv'}    ({len(long_df)} rows)")
    report.append(f"Wrote {out/'diabetes_glucose.csv'} ({len(glucose_df)} rows)")

    if not args.no_excel:
        xlsx_path = out / "diabetes_dataset.xlsx"
        write_excel(xlsx_path, long_df, glucose_df)
        report.append(f"Wrote {xlsx_path} (sheets: glucose_table, event_log)")

    (out / "data_quality_report.txt").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
