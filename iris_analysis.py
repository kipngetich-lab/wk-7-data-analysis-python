# IRIS DATASET ANALYSIS AND VISUALIZATION

import pandas as pd
import numpy as np # For potentially introducing NaNs for demonstration
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# --- Configuration ---
CSV_FILE_NAME = 'iris_dataset.csv'
CREATE_CSV_FROM_SKLEARN = True # Set to False if you already have iris_dataset.csv

def load_and_prepare_dataset(file_path, create_from_sklearn=False):
    """
    Loads the Iris dataset.
    If create_from_sklearn is True, it loads from sklearn, saves to CSV, then reloads.
    Otherwise, it tries to load directly from the CSV file_path.
    """
    df = None
    if create_from_sklearn:
        print(f"Loading Iris dataset from sklearn and saving to '{file_path}'...")
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        # Convert target numbers to species names
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        try:
            df.to_csv(file_path, index=False)
            print(f"Dataset saved to '{file_path}'.")
        except Exception as e:
            print(f"Error saving dataset to CSV: {e}")
            # Continue with the df loaded from sklearn if saving fails
            if df is None: # Should not happen if sklearn load worked
                 return None

    # Try loading from CSV (either freshly created or existing)
    print(f"\nAttempting to load dataset from '{file_path}'...")
    try:
        df_from_csv = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from '{file_path}'.")
        return df_from_csv
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        if df is not None: # If we had it from sklearn and saving failed but csv loading also failed
            print("Returning dataset loaded directly from sklearn as fallback.")
            return df
        return None
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        if df is not None:
            print("Returning dataset loaded directly from sklearn as fallback.")
            return df
        return None


def explore_dataset(df):
    """Explores the dataset structure, data types, and missing values."""
    print("\n--- Task 1: Load and Explore the Dataset ---")
    if df is None:
        print("Dataset not loaded. Skipping exploration.")
        return

    print("\n1. First 5 rows of the dataset (df.head()):")
    print(df.head())

    print("\n2. Dataset structure and data types (df.info()):")
    df.info()

    print("\n3. Check for missing values (df.isnull().sum()):")
    missing_values = df.isnull().sum()
    print(missing_values)

    if missing_values.sum() == 0:
        print("No missing values found.")
    else:
        print("Missing values found. Proceeding to clean data...")


def clean_dataset(df):
    """Cleans the dataset by handling missing values."""
    if df is None:
        print("Dataset not loaded. Skipping cleaning.")
        return df

    print("\n--- Data Cleaning ---")
    # For demonstration, let's introduce a few NaNs if none exist
    # In a real scenario, you'd handle existing NaNs
    if df.isnull().sum().sum() == 0:
        print("No missing values to clean initially. Introducing some for demonstration.")
        # Introduce NaNs into a numerical column, e.g., 'sepal length (cm)'
        # Make sure the column exists before trying to modify it
        if 'sepal length (cm)' in df.columns:
            num_nans = min(5, len(df) // 10) # Introduce NaNs in up to 5 rows or 10% of data
            if num_nans > 0:
                nan_indices = np.random.choice(df.index, num_nans, replace=False)
                df.loc[nan_indices, 'sepal length (cm)'] = np.nan
                print(f"Introduced {df['sepal length (cm)'].isnull().sum()} NaN(s) in 'sepal length (cm)' for demonstration.")
        else:
            print("Column 'sepal length (cm)' not found for introducing NaNs.")

    # Handle missing values
    # Option 1: Fill numerical NaNs with the mean
    numerical_cols = df.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in '{col}' with its mean ({mean_val:.2f}).")

    # Option 2: Drop rows with any remaining NaNs (e.g., if categorical had NaNs)
    # df.dropna(inplace=True)
    # print("Dropped rows with any remaining NAs.")

    print("\nMissing values after cleaning (df.isnull().sum()):")
    print(df.isnull().sum())
    print("Data cleaning complete.")
    return df


def basic_data_analysis(df):
    """Performs basic data analysis."""
    print("\n--- Task 2: Basic Data Analysis ---")
    if df is None:
        print("Dataset not loaded. Skipping analysis.")
        return

    # Ensure numerical_cols are correctly identified after potential cleaning
    numerical_cols_for_describe = df.select_dtypes(include=np.number).columns

    print("\n1. Basic statistics of numerical columns (df.describe()):")
    # Exclude 'species' if it was numerically encoded and not handled as categorical
    # In our case, 'species' is categorical, but describe() only works on numerical.
    if not numerical_cols_for_describe.empty:
        print(df[numerical_cols_for_describe].describe())
    else:
        print("No numerical columns found to describe.")


    print("\n2. Grouping by 'species' and computing mean of numerical columns:")
    if 'species' in df.columns and not numerical_cols_for_describe.empty:
        # Select only numerical columns for aggregation
        grouped_analysis = df.groupby('species')[numerical_cols_for_describe].mean()
        print(grouped_analysis)

        print("\nPatterns/Findings from Grouping:")
        print("- Setosa species generally has smaller sepal and petal dimensions compared to Versicolor and Virginica.")
        print("- Virginica species tends to have the largest petal length and width.")
        print("- Sepal width seems less distinct between Versicolor and Virginica compared to other features.")
    else:
        print("Could not perform grouped analysis: 'species' column or numerical columns missing.")


def data_visualization(df):
    """Creates and displays various data visualizations."""
    print("\n--- Task 3: Data Visualization ---")
    if df is None:
        print("Dataset not loaded. Skipping visualization.")
        return

    sns.set_theme(style="whitegrid") # Using seaborn for nicer default styles

    # 1. Line Chart (Trend of 'sepal length (cm)' over the dataset index)
    # Iris dataset doesn't have a natural time series. We'll plot a feature against its index.
    if 'sepal length (cm)' in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['sepal length (cm)'], marker='o', linestyle='-', color='b', label='Sepal Length (cm)')
        plt.title('Trend of Sepal Length (cm) over Dataset Index')
        plt.xlabel('Data Point Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Column 'sepal length (cm)' not found for line chart.")


    # 2. Bar Chart (Average petal length per species)
    if 'species' in df.columns and 'petal length (cm)' in df.columns:
        avg_petal_length = df.groupby('species')['petal length (cm)'].mean().reset_index()
        plt.figure(figsize=(8, 6))
        sns.barplot(x='species', y='petal length (cm)', data=avg_petal_length, palette='viridis')
        plt.title('Average Petal Length per Species')
        plt.xlabel('Species')
        plt.ylabel('Average Petal Length (cm)')
        plt.show()
    else:
        print("Columns 'species' or 'petal length (cm)' not found for bar chart.")

    # 3. Histogram (Distribution of 'petal width (cm)')
    if 'petal width (cm)' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['petal width (cm)'], kde=True, bins=15, color='purple')
        plt.title('Distribution of Petal Width (cm)')
        plt.xlabel('Petal Width (cm)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("Column 'petal width (cm)' not found for histogram.")

    # 4. Scatter Plot (Sepal length vs. Petal length, colored by species)
    if 'sepal length (cm)' in df.columns and 'petal length (cm)' in df.columns and 'species' in df.columns:
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, palette='bright', s=100) # s is marker size
        plt.title('Sepal Length vs. Petal Length by Species')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.legend(title='Species')
        plt.show()
    else:
        print("Required columns for scatter plot not found.")

    print("\nVisualizations generated.")
    print("Observations from Visualizations:")
    print("- The line chart shows variability in sepal length across the dataset samples.")
    print("- The bar chart clearly confirms that Virginica has the largest average petal length, followed by Versicolor, and then Setosa with the smallest.")
    print("- The histogram for petal width shows a somewhat multi-modal distribution, likely corresponding to the different species.")
    print("- The scatter plot effectively distinguishes the three species based on sepal and petal length. Setosa forms a distinct cluster, while Versicolor and Virginica show some overlap but are generally separable.")


def main():
    """Main function to run the data analysis pipeline."""
    print("--- Iris Dataset Analysis Assignment ---")

    # Task 1: Load and Explore
    iris_df = load_and_prepare_dataset(CSV_FILE_NAME, create_from_sklearn=CREATE_CSV_FROM_SKLEARN)
    if iris_df is None:
        print("\nFailed to load dataset. Exiting.")
        return

    explore_dataset(iris_df)

    # Task 1 (continued): Clean
    iris_df_cleaned = clean_dataset(iris_df.copy()) # Work on a copy for cleaning
    if iris_df_cleaned is None:
        print("\nDataset cleaning failed or was skipped. Exiting.")
        return

    # Task 2: Basic Data Analysis
    basic_data_analysis(iris_df_cleaned)

    # Task 3: Data Visualization
    data_visualization(iris_df_cleaned)

    print("\n--- Assignment Complete ---")
    print("\nOverall Findings/Observations Summary:")
    print("1. The Iris dataset consists of 150 samples from three species (Setosa, Versicolor, Virginica), with four features each.")
    print("2. The dataset (when loaded from sklearn) is clean, but we demonstrated handling for potential missing values.")
    print("3. Statistical analysis and grouping show distinct differences in feature measurements across the species, particularly in petal dimensions.")
    print("4. Visualizations (bar chart, scatter plot) strongly reinforce these differences, highlighting how features like petal length and width can be used to distinguish between the species.")
    print("5. Setosa is generally the most distinct species based on the measured features.")

if __name__ == "__main__":
    main()