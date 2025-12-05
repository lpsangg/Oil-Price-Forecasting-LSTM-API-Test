import pandas as pd
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
import matplotlib.pyplot as plt


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV file."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_continuous_series(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to continuous daily time series."""
    df = df.copy()
    df = df.set_index("date")
    daily_df = df.resample("D").asfreq()
    return daily_df


def fill_missing_with_rolling(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    """Fill missing values using centered rolling average."""
    df = df.copy()

    # tính rolling
    rolling = df["value"].rolling(
        window=window_size * 2 + 1,
        min_periods=1,
        center=True
    ).mean()

    # KHÔNG dùng inplace để tránh FutureWarning
    df["value"] = df["value"].fillna(rolling)

    return df


def preprocess_data(csv_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline."""
    df = load_raw_data(csv_path)
    df = create_continuous_series(df)
    df = fill_missing_with_rolling(df, window_size=5)

    # reset index + thêm Year
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"])
    df["Year"] = df["date"].dt.year
    df = df.sort_values("date")

    return df


def plot_series(df: pd.DataFrame):
    """Plot series (optional, not used in unit test)."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["value"], label="Value")
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Average Oil Price of OPEC Member Countries")
    plt.legend()

    years = YearLocator()
    years_fmt = DateFormatter("%Y")
    months = MonthLocator()

    ax = plt.gca()
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(months)

    plt.tight_layout()
    plt.show()


def save_preprocessed(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, index=False)


# --------------------------------------------------------------
# Main execution block (script mode)
# --------------------------------------------------------------
if __name__ == "__main__":
    input_path = "./data/QDL-OPEC.csv"
    output_path = "./data/preprocess-QDL-OPEC.csv"

    df = preprocess_data(input_path)
    print(df.head())

    plot_series(df)
    save_preprocessed(df, output_path)
