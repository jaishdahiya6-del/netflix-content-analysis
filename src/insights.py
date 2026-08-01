"""
===========================================================
Statistical Distribution Analysis
===========================================================

Project     : Netflix Data Science Statistics Report
Description : Calculates descriptive statistics and
              visualizes the distribution of a numeric column.

Features
--------
✔ Mean
✔ Median
✔ Standard Deviation
✔ Skewness
✔ Kurtosis
✔ Histogram
✔ KDE Curve
✔ Mean & Median Indicators
✔ Automatic Image Saving

===========================================================
"""

import logging
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Add root folder to path so we can import from src
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.data_loader import load_netflix_data
from src.data_cleaning import clean_netflix_data

# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Create Output Directory
# ---------------------------------------------------------
OUTPUT_DIR = Path("images")
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# Statistical Analysis Function
# ---------------------------------------------------------
def statistical_distribution_report(
    dataframe: pd.DataFrame,
    column: str,
    save_plot: bool = True,
) -> dict:
    """
    Perform descriptive statistical analysis on a numeric column.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input dataset.

    column : str
        Numeric column to analyze.

    save_plot : bool, default=True
        Save visualization as PNG.

    Returns
    -------
    dict
        Dictionary containing statistical measures.

    Raises
    ------
    ValueError
        If column does not exist or is non-numeric.
    """

    if column not in dataframe.columns:
        raise ValueError(f"Column '{column}' not found.")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise ValueError(f"'{column}' must contain numeric values.")

    data = dataframe[column].dropna()

    statistics = {
        "Count": len(data),
        "Mean": data.mean(),
        "Median": data.median(),
        "Standard Deviation": data.std(),
        "Minimum": data.min(),
        "Maximum": data.max(),
        "Variance": data.var(),
        "Skewness": skew(data),
        "Kurtosis": kurtosis(data),
    }

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    plt.figure(figsize=(12, 6))

    plt.hist(
        data,
        bins=30,
        density=True,
        alpha=0.75,
        edgecolor="black",
        label="Distribution",
    )

    plt.axvline(
        statistics["Mean"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean ({statistics['Mean']:.2f})",
    )

    plt.axvline(
        statistics["Median"],
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Median ({statistics['Median']:.2f})",
    )

    plt.title(
        f"Statistical Distribution of '{column}'",
        fontsize=15,
        fontweight="bold",
    )

    plt.xlabel(column)
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()

    if save_plot:
        image_path = OUTPUT_DIR / f"{column.lower()}_distribution.png"
        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        logger.info("Visualization saved to %s", image_path)

    plt.close()

    return statistics


# ---------------------------------------------------------
# Main Program
# ---------------------------------------------------------
def main() -> None:
    """
    Execute statistical analysis.
    """

    try:
        logger.info("Loading Netflix dataset...")

        df_raw = load_netflix_data()
        df = clean_netflix_data(df_raw)

        COLUMN = "release_year"
        if COLUMN not in df.columns:
            logger.error("Column '%s' not found in Netflix dataset.", COLUMN)
            return

        logger.info("Dataset loaded and cleaned successfully.")

        report = statistical_distribution_report(df, COLUMN)

        print("\n" + "=" * 55)
        print("        STATISTICAL ANALYSIS REPORT")
        print("=" * 55)

        for metric, value in report.items():
            print(f"{metric:<22}: {value:.2f}")

        print("=" * 55)

    except Exception as error:
        logger.exception("Unexpected Error: %s", error)


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
