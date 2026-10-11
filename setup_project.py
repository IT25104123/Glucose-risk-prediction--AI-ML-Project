
import subprocess
import sys
from pathlib import Path

FOLDERS = [
    "data/raw",
    "data/external",
    "results/eda_visualizations",
    "results/logs",
    "results/outputs",
]

RAW = Path("data/raw/Diabetes-Data")
OUT = Path("results/outputs")


def step(n, msg):
    print(f"\n[{n}] {msg}")


def main():
    step(1, "Creating folder structure")
    for f in FOLDERS:
        Path(f).mkdir(parents=True, exist_ok=True)
        print(f"    {f}")

    step(2, "Checking for the raw data")
    if not RAW.exists():
        sys.exit(
            f"    NOT FOUND: {RAW}\n"
            f"    Unzip Diabetes-Data-raw.zip so that data/raw/Diabetes-Data/data-01 exists,\n"
            f"    then run this script again."
        )
    n_files = len(list(RAW.glob("data-*")))
    print(f"    Found {n_files} patient files in {RAW}")
    if n_files != 70:
        print(f"    WARNING: expected 70 patient files, found {n_files}")

    step(3, "Building the all-records dataset (every raw row, nothing dropped)")
    subprocess.run(
        [sys.executable, "build_full_csv.py", "--input", str(RAW), "--output", str(OUT)],
        check=True,
    )

    step(4, "Building the modelling table (one row per glucose reading)")
    subprocess.run(
        [sys.executable, "prepare_diabetes_dataset.py",
         "--input", str(RAW), "--output", str(OUT)],
        check=True,
    )

    step(5, "Verifying")
    import pandas as pd

    all_df = pd.read_csv(OUT / "diabetes_all_records.csv")
    glu_df = pd.read_csv(OUT / "diabetes_glucose.csv")

    checks = [
        ("diabetes_all_records.csv rows", len(all_df), 29330),
        ("diabetes_all_records.csv cols", all_df.shape[1], 19),
        ("patients", all_df.patient_id.nunique(), 70),
        ("diabetes_glucose.csv rows", len(glu_df), 13511),
        ("diabetes_glucose.csv cols", glu_df.shape[1], 31),
    ]
    ok = True
    for name, got, expected in checks:
        mark = "OK " if got == expected else "XX "
        if got != expected:
            ok = False
        print(f"    {mark} {name}: {got} (expected {expected})")

    print("\n" + ("Setup complete. Open the notebook and select the .venv kernel."
                  if ok else
                  "Something did not match. Check the messages above."))


if __name__ == "__main__":
    main()
