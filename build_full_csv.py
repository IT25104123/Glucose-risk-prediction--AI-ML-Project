import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CODE_MAP = {
    33: ("Regular insulin dose",                  "insulin",  "regular"),
    34: ("NPH insulin dose",                      "insulin",  "nph"),
    35: ("UltraLente insulin dose",               "insulin",  "ultralente"),
    48: ("Unspecified blood glucose measurement", "glucose",  "unspecified"),
    56: ("Blood glucose (UNDOCUMENTED code 56)",  "glucose",  "undocumented_56"),
    57: ("Unspecified blood glucose measurement", "glucose",  "unspecified"),
    58: ("Pre-breakfast blood glucose",           "glucose",  "pre_breakfast"),
    59: ("Post-breakfast blood glucose",          "glucose",  "post_breakfast"),
    60: ("Pre-lunch blood glucose",               "glucose",  "pre_lunch"),
    61: ("Post-lunch blood glucose",              "glucose",  "post_lunch"),
    62: ("Pre-supper blood glucose",              "glucose",  "pre_supper"),
    63: ("Post-supper blood glucose",             "glucose",  "post_supper"),
    64: ("Pre-snack blood glucose",               "glucose",  "pre_snack"),
    65: ("Hypoglycemic symptoms",                 "event",    "hypo_symptoms"),
    66: ("Typical meal ingestion",                "meal",     "typical"),
    67: ("More-than-usual meal ingestion",        "meal",     "more"),
    68: ("Less-than-usual meal ingestion",        "meal",     "less"),
    69: ("Typical exercise activity",             "exercise", "typical"),
    70: ("More-than-usual exercise activity",     "exercise", "more"),
    71: ("Less-than-usual exercise activity",     "exercise", "less"),
    72: ("Unspecified special event",             "event",    "special"),
}

GLUCOSE_MIN, GLUCOSE_MAX = 20.0, 500.0
INSULIN_MIN, INSULIN_MAX = 0.0, 100.0
HYPO_THRESHOLD, HYPER_THRESHOLD = 70.0, 180.0


def to_number(raw):
    if raw is None:
        return np.nan
    s = str(raw).strip().strip("'\"")
    if s == "":
        return np.nan
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Diabetes-Data")
    ap.add_argument("--output", default="data")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input).iterdir()
                   if p.is_file() and re.match(r"^data-\d+$", p.name))

    rows = []
    for path in files:
        pid = int(re.search(r"(\d+)", path.name).group(1))
        with open(path, "r", errors="ignore") as fh:
            for line_no, line in enumerate(fh, start=1):
                if line.strip() == "":
                    continue
                # keep positions: do NOT collapse empty fields, so a missing
                # date stays visible as an empty first field
                parts = line.rstrip("\n").split("\t")
                parts += [""] * (4 - len(parts))
                rows.append({
                    "patient_id": pid,
                    "source_file": path.name,
                    "line_no": line_no,
                    "raw_date": parts[0].strip(),
                    "raw_time": parts[1].strip(),
                    "raw_code": parts[2].strip(),
                    "raw_value": parts[3].strip(),
                })

    df = pd.DataFrame(rows)
    df.insert(0, "row_id", range(1, len(df) + 1))

    # ---------- parse ----------
    df["date"] = pd.to_datetime(df["raw_date"], format="%m-%d-%Y", errors="coerce")

    tm = df["raw_time"].str.extract(r"^(\d{1,2}):(\d{1,2})$")
    hh = pd.to_numeric(tm[0], errors="coerce")
    mm = pd.to_numeric(tm[1], errors="coerce")
    ok_time = hh.between(0, 23) & mm.between(0, 59)
    df["time"] = np.where(
        ok_time,
        hh.fillna(0).astype(int).astype(str).str.zfill(2) + ":" +
        mm.fillna(0).astype(int).astype(str).str.zfill(2),
        np.nan,
    )

    df["timestamp"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"], errors="coerce"
    )

    df["code"] = pd.to_numeric(df["raw_code"], errors="coerce").astype("Int64")
    df["value"] = df["raw_value"].map(to_number)

    fallback = ("Unknown / undocumented code", "unknown", "unknown")
    df["code_description"] = df["code"].map(
        lambda c: CODE_MAP.get(c, fallback)[0] if pd.notna(c) else "Missing code")
    df["measurement_type"] = df["code"].map(
        lambda c: CODE_MAP.get(c, fallback)[1] if pd.notna(c) else "unknown")
    df["context"] = df["code"].map(
        lambda c: CODE_MAP.get(c, fallback)[2] if pd.notna(c) else "unknown")

    # ---------- risk band (glucose rows only) ----------
    def risk(r):
        if r["measurement_type"] != "glucose" or pd.isna(r["value"]):
            return np.nan
        v = r["value"]
        if v < HYPO_THRESHOLD:
            return "Hypoglycemic"
        if v > HYPER_THRESHOLD:
            return "Hyperglycemic"
        return "Normal"

    df["risk_class"] = df.apply(risk, axis=1)

    # ---------- issue flags (nothing is deleted) ----------
    issues = []
    for _, r in df.iterrows():
        problems = []
        if pd.isna(r["date"]):
            problems.append("bad_or_missing_date")
        if pd.isna(r["time"]):
            problems.append("bad_or_missing_time")
        if pd.isna(r["code"]):
            problems.append("missing_code")
        elif r["measurement_type"] == "unknown":
            problems.append("undocumented_code")
        if pd.isna(r["value"]) and r["raw_value"] != "":
            problems.append("non_numeric_value")
        if r["measurement_type"] == "glucose" and pd.notna(r["value"]) \
                and not (GLUCOSE_MIN <= r["value"] <= GLUCOSE_MAX):
            problems.append("glucose_out_of_range")
        if r["measurement_type"] == "insulin" and pd.notna(r["value"]) \
                and not (INSULIN_MIN <= r["value"] <= INSULIN_MAX):
            problems.append("insulin_out_of_range")
        issues.append(";".join(problems))
    df["issue"] = issues

    dup = df.duplicated(subset=["patient_id", "timestamp", "code", "value"], keep="first")
    df.loc[dup & df["timestamp"].notna(), "issue"] = (
        df.loc[dup & df["timestamp"].notna(), "issue"]
        .apply(lambda s: (s + ";duplicate_record").strip(";"))
    )

    df["is_clean"] = df["issue"] == ""

    cols = ["row_id", "patient_id", "source_file", "line_no",
            "raw_date", "raw_time", "raw_code", "raw_value",
            "date", "time", "timestamp",
            "code", "code_description", "measurement_type", "context",
            "value", "risk_class", "issue", "is_clean"]
    df = df[cols]
    df["date"] = df["date"].dt.date

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "diabetes_all_records.csv"
    df.to_csv(target, index=False, encoding="utf-8-sig")

    print(f"Wrote {target}")
    print(f"Total rows: {len(df)}  (every raw record, nothing dropped)")
    print(f"Clean rows: {df['is_clean'].sum()}   Flagged rows: {(~df['is_clean']).sum()}")
    print("\nIssue breakdown:")
    flagged = df.loc[~df["is_clean"], "issue"].str.split(";").explode()
    print(flagged.value_counts().to_string())
    print("\nRecord type breakdown:")
    print(df["measurement_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
