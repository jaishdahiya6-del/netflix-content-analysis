# run_project.py
# ─────────────────────────────────────────────────────────────────
# PURPOSE: One script to run the ENTIRE Netflix analysis project
# Just run: python run_project.py
# ─────────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader   import load_netflix_data # type: ignore
from data_cleaning import clean_netflix_data # type: ignore
from eda           import run_all_eda # type: ignore
from ml_models     import run_all_ml # type: ignore
from insights      import generate_insights # type: ignore

def main():
    print("\n" + "█"*60)
    print("█  NETFLIX CONTENT ANALYSIS — FULL PIPELINE             █")
    print("█"*60)

    # Step 1: Load
    print("\n[1/4] Loading dataset...")
    df_raw = load_netflix_data()

    # Step 2: Clean
    print("\n[2/4] Cleaning data...")
    df_clean = clean_netflix_data(df_raw)

    # Step 3: EDA (8 charts)
    print("\n[3/4] Running EDA — generating 8 charts...")
    run_all_eda(df_clean)

    # Step 4: ML (3 models + 4 ML charts)
    print("\n[4/4] Running Machine Learning models...")
    run_all_ml(df_clean)

    # Step 5: Business Insights
    print("\n[5/5] Generating business insights report...")
    generate_insights(df_clean)

    print("\n" + "█"*60)
    print("█  PROJECT COMPLETE!                                     █")
    print("█  Charts  → /visualizations/ (12 PNG files)            █")
    print("█  Report  → /reports/business_insights_report.txt      █")
    print("█"*60)

if __name__ == "__main__":
    main()