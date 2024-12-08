import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip().str.lower()
        
        if 'timestamp' not in df.columns:
            raise ValueError(f"Error: 'timestamp' column not found. Available columns: {df.columns}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data from {filepath}: {e}")
        return pd.DataFrame()



def clean_data(df: pd.DataFrame, output_path: str = "app/datasets/cleaned_data.csv") -> pd.DataFrame:
    """
    Cleans the given DataFrame and saves the cleaned data to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        output_path (str): The file path to save the cleaned data.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    try:
        # Drop entirely null columns and duplicates
        df = df.dropna(axis=1, how='all')
        df = df.drop_duplicates()

        # Fill missing values with mean for key columns
        numeric_cols = ['ghi', 'dni', 'dhi', 'moda', 'modb', 'tamb', 'rh', 'ws']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # Clip negative values
        for col in ['ghi', 'dni', 'dhi', 'moda', 'modb']:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        # Handle outliers for multiple columns
        for col in ['ghi', 'dni', 'dhi', 'moda', 'modb']:
            if col in df.columns:
                df[f'{col}_zscore'] = zscore(df[col])
                df = df[df[f'{col}_zscore'].abs() < 4]
                df.drop(f'{col}_zscore', axis=1, inplace=True)

        # Save the cleaned data to a CSV file
        df.to_csv(output_path, index=False)
        print(f"Data cleaned and saved to {output_path}. {df.shape[0]} rows remaining.")

        return df
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    try:
        summary_stats = df.select_dtypes(include=[np.number]).describe().T
        print(summary_stats)
        return summary_stats
    except Exception as e:
        print(f"Error generating summary statistics: {e}")
        return pd.DataFrame()


def plot_correlation_heatmap(df: pd.DataFrame):
    try:
        plt.figure(figsize=(12, 10))
        corr = df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap')
        plt.show()
    except Exception as e:
        print(f"Error plotting heatmap: {e}")


def plot_time_series(df: pd.DataFrame, column: str):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df[column])
        plt.xlabel('Time')
        plt.ylabel(column.capitalize())
        plt.title(f'{column.capitalize()} Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting time series for {column}: {e}")


def plot_time_series_with_precipitation(df: pd.DataFrame):
    try:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('GHI (W/m²)', color='tab:blue')
        ax1.plot(df['timestamp'], df['ghi'], color='tab:blue', label='GHI')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Precipitation (mm/min)', color='tab:green')
        ax2.plot(df['timestamp'], df['precipitation'], color='tab:green', label='Precipitation')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        
        fig.tight_layout()
        plt.title('Time Series of GHI and Precipitation')
        plt.show()
    except Exception as e:
        print(f"Error plotting GHI and precipitation: {e}")


def plot_wind_rose(df: pd.DataFrame):
    try:
        from windrose import WindroseAxes
        ax = WindroseAxes.from_ax()
        ax.bar(df['wd'], df['ws'], normed=True, opening=0.8, edgecolor='white')
        ax.set_legend()
        plt.title('Wind Rose (WD vs. WS)')
        plt.show()
    except Exception as e:
        print(f"Error plotting wind rose: {e}")


def evaluate_cleaning_impact(df: pd.DataFrame):
    try:
        cleaned = df[df['cleaning'] == 1]
        not_cleaned = df[df['cleaning'] == 0]
        
        metrics = ['moda', 'modb']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x='cleaning', y=metric, palette='Set2')
            plt.title(f'Impact of Cleaning on {metric.upper()}')
            plt.xlabel('Cleaning (1 = Yes, 0 = No)')
            plt.ylabel(metric.upper())
            plt.show()
        
        print("Average sensor readings before and after cleaning:")
        for metric in metrics:
            mean_cleaned = cleaned[metric].mean()
            mean_not_cleaned = not_cleaned[metric].mean()
            print(f"{metric.upper()}: Cleaned = {mean_cleaned:.2f}, Not Cleaned = {mean_not_cleaned:.2f}")
    except Exception as e:
        print(f"Error evaluating cleaning impact: {e}")


def plot_bubble_chart(df: pd.DataFrame):
    try:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='ghi', y='tamb', size='ws', hue='rh', data=df, sizes=(20, 200), palette='viridis')
        plt.xlabel('GHI')
        plt.ylabel('Tamb')
        plt.title('Bubble Chart: GHI vs Tamb vs WS (Bubble Size) with RH as Hue')
        plt.legend(loc='upper right')
        plt.show()
    except Exception as e:
        print(f"Error plotting bubble chart: {e}")
