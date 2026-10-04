"""
Generate a SYNTHETIC dataset for the everolimus liver transplant analytics notebook.

All values are random numbers drawn from made-up distributions. No patient-level
record was copied, sampled or used to fit them; the distribution parameters are
rough, invented values chosen only so the notebook runs and the charts look
plausible. Results from this file illustrate how the notebook works and say
nothing about real patients.

Usage:
    python generate_synthetic_data.py
Output:
    data/Liver_Transplant_Rejection.xlsx
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 40  # number of synthetic patients
rng = np.random.default_rng(SEED)

TP = ["DAY 7", "DAY 30", "DAY 45", "DAY 60", "DAY 90", "MONTH 6", "MONTH 12", "MONTH 18"]
TP_EVER = ["DAY7"] + TP[1:]  # the notebook expects the label 'DAY7' for everolimus levels
P_MISSING = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60, 0.80])
RESP_FACTOR = np.array([0.85, 0.70, 0.60, 0.50, 0.45, 0.40, 0.40, 0.40])
NONRESP_FACTOR = np.array([1.00, 0.95, 0.90, 0.90, 0.95, 1.00, 1.00, 1.00])

# ---- patient-level attributes -------------------------------------------------
response = rng.choice([1, 2], size=N, p=[0.6, 0.4])        # 1 = responder, 2 = non-responder
age = rng.integers(25, 72, size=N)
gender = rng.choice([1, 2], size=N, p=[0.75, 0.25])        # 1 = male, 2 = female
hospital = rng.choice([1, 2], size=N)
tx_type = rng.choice([1, 2], size=N, p=[0.7, 0.3])
biopsy_full = rng.choice([1, 2, 3, 4, 5, 6, 7, 0], size=N, p=[0.35, 0.05, 0.3, 0.1, 0.05, 0.05, 0.05, 0.05])
biopsy_initial = rng.choice([1, 2, 3], size=N, p=[0.5, 0.35, 0.15])
days_rej_to_ever = rng.integers(0, 400, size=N)            # always >= 0 in the synthetic data
days_tx_to_ever = days_rej_to_ever + rng.integers(30, 1500, size=N)
indication = rng.choice([1, 2], size=N, p=[0.8, 0.2])
status = rng.choice([1, 2, 3], size=N, p=[0.7, 0.2, 0.1])


def trajectory(base_mean, base_sd, nonresp_uplift=1.0, floor=0.05, decimals=2):
    """Return (baseline, matrix N x 8) of synthetic lab values with a response-dependent trend."""
    base = np.clip(rng.normal(base_mean, base_sd, N), floor, None)
    base = np.where(response == 2, base * nonresp_uplift, base)
    factor = np.where(response[:, None] == 1, RESP_FACTOR[None, :], NONRESP_FACTOR[None, :])
    values = base[:, None] * factor * rng.lognormal(0, 0.2, (N, len(TP)))
    values = np.round(np.clip(values, floor, None), decimals)
    mask = rng.random((N, len(TP))) < P_MISSING[None, :]
    values = values.astype(float)
    values[mask] = np.nan
    return np.round(base, decimals), values


markers = {
    # name: (mean, sd, non-responder uplift, floor, decimals)
    "TOTAL BILIRUBIN": (8, 4, 1.6, 0.2, 2),
    "DIRECT BILIRUBIN": (5, 3, 1.6, 0.1, 2),
    "ALP": (300, 90, 1.1, 30, 0),
    "AST": (120, 50, 1.1, 10, 0),
    "ALT": (140, 60, 1.1, 10, 0),
    "GGT": (350, 120, 1.1, 10, 0),
    "CR": (1.1, 0.3, 1.2, 0.3, 2),
    "ALBUMIN": (3.6, 0.4, 0.95, 1.5, 2),
    "TAC LEVELS": (6, 1.5, 1.0, 1.0, 1),
    "EVEROLIMUS LEVELS": (4.5, 1.2, 1.0, 0.5, 1),
}

baseline = {}
raw = {}
for name, (m, sd, up, fl, dec) in markers.items():
    b, v = trajectory(m, sd, up, fl, dec)
    baseline[name] = b
    cols = TP_EVER if name == "EVEROLIMUS LEVELS" else TP
    for j, tp in enumerate(cols):
        raw[(name, tp)] = v[:, j]


def ptinr_strings(n):
    inr = np.round(np.clip(rng.normal(1.3, 0.35, n), 0.8, 3.5), 2)
    pt = np.round(inr * 11 + rng.normal(0, 1, n), 1)
    return [f"{p}/{i}" for p, i in zip(pt, inr)]


# PT/INR as 'pt/inr' text (the notebook splits on '/')
for tp in TP:
    vals = np.array(ptinr_strings(N), dtype=object)
    vals[rng.random(N) < P_MISSING[TP.index(tp)]] = np.nan
    raw[("PT/INR", tp)] = vals
pre = np.array(ptinr_strings(N), dtype=object)
raw[("PRE-EVEROLIMUS VALUES", "PT/INR")] = pre

# most recent follow-up values
recent_map = {"T.B": "TOTAL BILIRUBIN", "D.B": "DIRECT BILIRUBIN", "ALP": "ALP", "AST": "AST", "ALT": "ALT",
              "GGT": "GGT", "CR": "CR", "ALB": "ALBUMIN", "TAC LEVEL": "TAC LEVELS"}
RECENT = "Last Date of Visit/Recent Follow up Lab values"
for short, full in recent_map.items():
    m, sd, up, fl, dec = markers[full]
    b, v = trajectory(m, sd, up, fl, dec)
    raw[(RECENT, short)] = v[:, 5]
vals = np.array(ptinr_strings(N), dtype=object)
vals[rng.random(N) < 0.3] = np.nan
raw[(RECENT, "PT/INR")] = vals

raw_df = pd.DataFrame(raw)

# ---- 'FINAL CODED SHEET' (one header row at row index 2, normalised by the notebook) ----
coded = pd.DataFrame({
    "S.NO": np.arange(1, N + 1),
    "TX - HOSPITAL ( INHOUSE = 1, OUTSIDE = 2)": hospital,
    "AGE": age,
    "GENDER (MALE = 1, FEMALE = 2)": gender,
    "TYPEOFTX (LDLT = 1, DDLT = 2)": tx_type,
    "BIOPSY RESULTS (CHRONIC REJECTION = 1, ACUTE CELLULAR REJECTION = 2, T CELL MEDIATED REJECTION = 3, "
    "CHRONIC DUCTOPENIC REJECTION = 4, LOBULAR HEPATITIS = 5, BILIRUBINOSTATIS = 6, RECURRENCE = 7, NO REJECTION = 0)": biopsy_full,
    "BIOPSY RESULTS (CHRONIC REJECTION = 1, T CELL MEDIATED REJECTION = 2, CHRONIC DUCTOPENIC REJECTION = 3)": biopsy_initial,
    "REMARKS (RESPONDERS = 1, NON RESPONDERS= 2)": response,
    "DURATION BETWEEN TX AND STARTING EVEROLIMUS(IN DAYS)": days_tx_to_ever,
    "DURATION BETWEEN TX AND STARTING EVEROLIMUS(IN MONTHS)": np.round(days_tx_to_ever / 30.44, 1),
    "DURATION BETWEEN REJECTION AND STARTING OF EVEROLIMUS IN DAYS": days_rej_to_ever,
    "DURATION BETWEEN REJECTION AND STARTING OF EVEROLIMUS IN MONTHS": np.round(days_rej_to_ever / 30.44, 2),
    "T.B": baseline["TOTAL BILIRUBIN"],
    "AST": baseline["AST"],
    "ALT": baseline["ALT"],
    "ALP": baseline["ALP"],
    "GGT": baseline["GGT"],
    "ALBUMIN": baseline["ALBUMIN"],
    "CR": baseline["CR"],
    "TAC LEVEL": baseline["TAC LEVELS"],
    "INDICATION OF STARTING EVEROLIMUS (REJECTION = 1, HCC, REJECTION =2)": indication,
    "MEDICATION STATUS ( ON EVEROLIMUS = 1, STOPPED = 2, DEAD = 3)": status,
})
# a few malformed entries, like real spreadsheets, so the notebook's cleaning step has work to do
for col in ["AST", "ALT", "GGT"]:
    idx = rng.choice(N, size=2, replace=False)
    coded[col] = coded[col].astype(object)
    coded.loc[idx, col] = rng.choice(["-", "NA", "nil"], size=2)

# ---- write the workbook ------------------------------------------------------------
os.makedirs("data", exist_ok=True)
path = os.path.join("data", "Liver_Transplant_Rejection.xlsx")
with pd.ExcelWriter(path, engine="openpyxl") as xl:
    # first sheet: the notebook reads it with header=[1, 2] (row 0 blank, row 1 group, row 2 timepoint)
    ncols = raw_df.shape[1]
    header_group = [c[0] for c in raw_df.columns]
    header_time = [c[1] for c in raw_df.columns]
    body = raw_df.astype(object).where(raw_df.notna(), None).values.tolist()
    sheet = pd.DataFrame([[None] * ncols, header_group, header_time] + body)
    sheet.to_excel(xl, sheet_name="RAW LONGITUDINAL", index=False, header=False)
    # coded sheet: notebook reads sheet_name='FINAL CODED SHEET', header=2
    coded.to_excel(xl, sheet_name="FINAL CODED SHEET", index=False, startrow=2)
print(f"Wrote {path} with {N} synthetic patients (seed {SEED}).")
