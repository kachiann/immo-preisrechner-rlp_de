"""
setup_db.py — Load German ImmobilienScout24 data into SQLite,
              filtered to Rhineland-Palatinate (Rheinland-Pfalz).

Files needed in data/:
  - apr20_price.csv      → rename to immo_data.csv
  - zip_lat_lang.csv     → keep as-is (used for map coordinates)

Both are from:
  https://www.kaggle.com/datasets/phanindraparashar/germany-housing-rent-and-price-data-set-apr-20

Usage:
    python setup_db.py
    python setup_db.py --all-germany
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data")
CSV_PATH    = DATA_DIR / "immo_data.csv"
ZIP_PATH    = DATA_DIR / "zip_lat_lang.csv"
DB_PATH     = DATA_DIR / "housing.db"

RP_STATE    = "Rheinland_Pfalz"   # underscore as it appears in obj_regio1
NEIGHBOURS  = [
    "Saarland", "Hessen",
    "Nordrhein_Westfalen", "Baden_Wuerttemberg"
]

# Columns to keep from the 83-column raw file
KEEP_COLS = [
    # Location
    "obj_regio1", "obj_regio2", "obj_regio3",
    "obj_zipCode",
    # Size
    "obj_livingSpace",
    "obj_noRooms",
    "obj_lotArea",
    "obj_usableArea",
    "obj_numberOfFloors",
    # Property details
    "obj_yearConstructed",
    "obj_lastRefurbish",
    "obj_constructionPhase",
    "obj_buildingType",
    "obj_heatingType",
    "obj_firingTypes",
    "obj_condition",
    "obj_interiorQual",
    "obj_energyEfficiencyClass",
    "obj_thermalChar",
    # Amenities (boolean)
    "obj_newlyConst",
    "obj_cellar",
    "obj_barrierFree",
    "obj_rented",
    "obj_noParkSpaces",
    # Market context
    "obj_pricetrendbuy",
    "obj_pricetrend",
    # Target
    "obj_purchasePrice",
]


def check_files() -> None:
    missing = []
    if not CSV_PATH.exists():
        missing.append(f"  • {CSV_PATH}  ← rename apr20_price.csv to immo_data.csv")
    if not ZIP_PATH.exists():
        missing.append(f"  • {ZIP_PATH}  ← download zip_lat_lang.csv from Kaggle")
    if missing:
        raise FileNotFoundError(
            "\n❌ Missing files:\n" + "\n".join(missing) +
            "\n\nDownload from:\n"
            "  https://www.kaggle.com/datasets/phanindraparashar/"
            "germany-housing-rent-and-price-data-set-apr-20"
        )


def load_zip_coords() -> pd.DataFrame:
    """Load zip → lat/lng lookup table."""
    zip_df = pd.read_csv(ZIP_PATH, low_memory=False)
    zip_df.columns = zip_df.columns.str.lower().str.strip()
    print(f"   zip_lat_lang.csv columns: {list(zip_df.columns)}")

    # Find the first column matching each role
    def find_col(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        # fallback: partial match
        for c in df.columns:
            if any(k in c for k in candidates):
                return c
        return None

    zip_col = find_col(zip_df, ["zipcode", "zip_code", "zip", "plz", "postleitzahl"])
    lat_col = find_col(zip_df, ["lat", "latitude"])
    lng_col = find_col(zip_df, ["lon", "lng", "long", "longitude"])

    if not all([zip_col, lat_col, lng_col]):
        raise ValueError(
            f"Could not identify zip/lat/lng columns in zip_lat_lang.csv.\n"
            f"Columns found: {list(zip_df.columns)}"
        )

    result = zip_df[[zip_col, lat_col, lng_col]].copy()
    result.columns = ["zip", "geo_lat", "geo_lng"]
    result["zip"] = result["zip"].astype(str).str.strip().str.zfill(5)
    result = result.dropna(subset=["geo_lat", "geo_lng"]).drop_duplicates("zip")
    print(f"   Loaded {len(result):,} zip codes with coordinates")
    return result


def load_csv() -> pd.DataFrame:
    print(f"📂 Reading {CSV_PATH} …")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"   Raw: {len(df):,} rows × {df.shape[1]} columns")
    return df


def filter_and_clean(df: pd.DataFrame, rp_only: bool, zip_df: pd.DataFrame) -> pd.DataFrame:
    # ── Keep only needed columns ───────────────────────────────────────────────
    available = [c for c in KEEP_COLS if c in df.columns]
    missing_c = set(KEEP_COLS) - set(available)
    if missing_c:
        print(f"   ⚠️  Columns not in file (will skip): {missing_c}")
    df = df[available].copy()

    # ── Regional filter ────────────────────────────────────────────────────────
    if rp_only and "obj_regio1" in df.columns:
        rp_df = df[df["obj_regio1"] == RP_STATE]
        if len(rp_df) < 300:
            print(
                f"   ⚠️  Only {len(rp_df)} RP rows found — adding neighbouring states…"
            )
            nb_df = df[df["obj_regio1"].isin(NEIGHBOURS)]
            df    = pd.concat([rp_df, nb_df], ignore_index=True)
            print(f"   Using {len(df):,} rows (RP + neighbours)")
        else:
            df = rp_df
            print(f"   Filtered to Rheinland-Pfalz: {len(df):,} rows")

    # ── Target: keep only rows with valid purchase price ──────────────────────
    if "obj_purchasePrice" in df.columns:
        before = len(df)
        df = df[df["obj_purchasePrice"].notna() & (df["obj_purchasePrice"] > 0)]
        # Realistic price range: €20k – €5M
        df = df[df["obj_purchasePrice"].between(20_000, 5_000_000)]
        print(f"   Valid price rows: {len(df):,}  (dropped {before - len(df):,})")

    # ── Clean living space: 15–1000 m² ────────────────────────────────────────
    if "obj_livingSpace" in df.columns:
        df = df[df["obj_livingSpace"].between(15, 1_000)]

    # ── Clean rooms: 1–20 ─────────────────────────────────────────────────────
    if "obj_noRooms" in df.columns:
        df = df[df["obj_noRooms"].between(1, 20)]

    # ── Clean lot area ────────────────────────────────────────────────────────
    if "obj_lotArea" in df.columns:
        df["obj_lotArea"] = df["obj_lotArea"].clip(upper=50_000)

    # ── Convert boolean columns ───────────────────────────────────────────────
    bool_cols = ["obj_newlyConst", "obj_cellar", "obj_barrierFree", "obj_rented"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = (
                df[col].map({"True": 1, "False": 0, True: 1, False: 0})
                .fillna(0).astype(int)
            )

    # ── Merge coordinates via zip code ────────────────────────────────────────
    if "obj_zipCode" in df.columns and zip_df is not None:
        df["zip_key"] = df["obj_zipCode"].astype(str).str.zfill(5).str[:5]
        df = df.merge(zip_df, left_on="zip_key", right_on="zip", how="left")
        df = df.drop(columns=["zip_key", "zip"], errors="ignore")
        matched = df["geo_lat"].notna().sum()
        print(f"   Coordinates matched: {matched:,} / {len(df):,} rows")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"   After cleaning: {len(df):,} rows")
    return df


def save_to_db(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("housing", conn, if_exists="replace", index=False)
    conn.close()

    target = "obj_purchasePrice"
    print(f"\n✅ Saved {len(df):,} records to {DB_PATH}")
    if target in df.columns:
        print(f"   Price range : €{df[target].min():,.0f} – €{df[target].max():,.0f}")
        print(f"   Median price: €{df[target].median():,.0f}")
    if "obj_regio1" in df.columns:
        print(f"   States      : {df['obj_regio1'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-germany", action="store_true",
                        help="Skip RP filter; train on full Germany")
    args = parser.parse_args()

    check_files()
    zip_df = load_zip_coords()
    df     = load_csv()
    df     = filter_and_clean(df, rp_only=not args.all_germany, zip_df=zip_df)
    save_to_db(df)
    print("\n🎉 Done!  Next step: python train.py")


if __name__ == "__main__":
    main()