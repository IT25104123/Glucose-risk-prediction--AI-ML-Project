
---
## Project overview

Predicting blood-glucose risk in insulin-dependent diabetes patients from
outpatient monitoring logs. The work combines a regression task (predict the
patient's next glucose reading) with a classification task (Hypoglycemic /
Normal / Hyperglycemic risk band).

##  Assigned dataset

**UCI Diabetes Data Set** (AAAI-94 Spring Symposium on Interpreting Clinical Data).

Credit: AIM-94 data set provided by Michael Kahn, MD, PhD, Washington
University, St. Louis, MO.

| Property | Value | 
| Patients | 70 (`data-01` … `data-70`) |
| Raw records | 29,330 |
| Date range | March 1988 – September 1991 |
| Fields per record | Date (MM-DD-YYYY), Time (HH:MM), Code, Value |
| Blood-glucose readings | 13,650 |
| Insulin dose records | 14,401 |

The raw files are a **long-format event log** with no header row. A single
`Value` column holds mg/dL for glucose codes and insulin units for dose codes,
so the code field must be decoded before any analysis is possible.


├── README.md
├── data/
│   ├── raw/                     # Diabetes-Data as provided (70 files + docs)
│   └── external/                # any external reference data
├── preprocessing_eda pipeline.ipynb
├── prepare_diabetes_dataset.py  # raw event log -> modelling table
├── build_full_csv.py            # raw event log -> single all-records csv
├── requirements.txt
└── results/
    ├── eda_visualizations/      # plots (PNG)
    ├── logs/                    # execution logs
    └── outputs/                 # processed datasets / features


##  How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python build_full_csv.py --input data/raw/Diabetes-Data --output results/outputs
python prepare_diabetes_dataset.py --input data/raw/Diabetes-Data --output results/outputs
```

Then open preprocessing_eda_pipeline.ipynb` and run all cells.

##. Preprocessing techniques implemented

| Technique | Where | Summary |
|  Handlinglassimbalance|notebook 1|
| Handling missing data | notebook §2 | |
| Encoding categorical variables | notebook §3 | |
| Outlier removal | notebook §4 | |
| Normalization / scaling | notebook §5 | |
| Feature engineering | notebook §6 | |

## 7. EDA visualizations

| # | Figure | File |
|---|---|---|
| 1 | | `results/eda_visualizations/` |
| 2 | | `results/eda_visualizations/` |
| 3 | | `results/eda_visualizations/` |

## 8. Known data quality issues

| Issue | Rows | Notes |
|---|---|---|
| Missing / malformed date | 40 | Empty first field shifts all others left (`data-27`, `data-29`) |
| Malformed time | 38 | Values such as `188:00`, `11:0`, `226` |
| Undocumented code 56 | 119 | Not in `Data-Codes`; value distribution matches blood glucose |
| Undocumented codes 0, 4, 36 | 35 | No discernible meaning |
| Duplicate records | 29 | Identical patient + timestamp + code + value |
| Glucose out of plausible range | 17 | Includes `0Hi` — censored, not missing |
| Insulin out of plausible range | 9 | |


requirements
@"
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.13
scikit-learn>=1.3
imbalanced-learn>=0.11
openpyxl>=3.1
jupyter>=1.0
ipykernel>=6.0
"@ | Out-File -Encoding utf8 requirements.txt