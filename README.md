# Iris Dataset Analysis and Visualization

This Python script performs a comprehensive analysis of the Iris dataset. It demonstrates loading data (from scikit-learn and CSV), exploring its structure, cleaning missing values (with a demonstration of introducing and handling NaNs), performing basic statistical analysis, and creating various visualizations using pandas, Matplotlib, and Seaborn. This project is designed to fulfill an assignment focusing on these core data science tasks.

## Features

*   **Data Loading:** Loads the Iris dataset from `sklearn.datasets` and demonstrates saving to/loading from a CSV file (`iris_dataset.csv`).
*   **Data Exploration:** Displays the first few rows, dataset info (data types, non-null counts), and checks for missing values.
*   **Data Cleaning:** Demonstrates handling missing values by filling numerical NaNs with the mean (includes a step to introduce NaNs for demonstration if the original dataset is clean).
*   **Basic Statistical Analysis:** Computes descriptive statistics for numerical features and performs group-by operations (e.g., mean feature values per species).
*   **Data Visualization:** Generates four types of plots:
    *   Line chart (trend of sepal length over the dataset index)
    *   Bar chart (average petal length per species)
    *   Histogram (distribution of petal width)
    *   Scatter plot (sepal length vs. petal length, colored by species)
*   **Customized Plots:** Utilizes Matplotlib and Seaborn for clear and informative visualizations with titles, labels, and legends.
*   **Error Handling:** Includes basic error handling for file operations (e.g., `FileNotFoundError`).
*   **Observations:** Prints findings and observations derived from the analysis and visualizations directly to the console.

## Prerequisites

*   Python 3.7+
*   The following Python libraries:
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`

## Installation

1. Ensure you have Python installed on your system.
2. Clone this repository or download the `iris_analysis.py` script.
3. Install the required libraries using pip. It's recommended to use a virtual environment:

    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv venv
    # Activate the virtual environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate

    # Install dependencies
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

## Usage

1. Navigate to the directory containing the `iris_analysis.py` script.
2. Run the script from your terminal:

    ```bash
    python iris_analysis.py
    ```

The script will perform the following actions:
*   If `CREATE_CSV_FROM_SKLEARN` is `True` (default), it will load the Iris dataset from scikit-learn, save it as `iris_dataset.csv`, and then load the data from this CSV file. If `CREATE_CSV_FROM_SKLEARN` is `False`, it will attempt to directly load `iris_dataset.csv`.
*   Print data exploration details (head, info, missing values) to the console.
*   Print data cleaning steps and results.
*   Print basic statistical analysis results (describe, grouped means).
*   Display the generated Matplotlib/Seaborn plots in separate windows one by one.
*   Print findings and observations to the console.

## Dataset

The script uses the **Iris flower dataset**.
*   Initially loaded via `sklearn.datasets.load_iris()`.
*   For the purpose of demonstrating CSV handling as per typical assignment requirements, it's then saved to `iris_dataset.csv` and reloaded from this CSV file.
*   The dataset contains 150 samples from three species of Iris flowers (Setosa, Versicolor, Virginica).
*   Four features are measured for each sample: sepal length, sepal width, petal length, and petal width (all in cm).

## Assignment Tasks Covered

This script is structured to address common data analysis assignment tasks:

*   **Task 1: Load and Explore the Dataset**
    *   Dataset loading (CSV format demonstrated).
    *   `df.head()` for initial inspection.
    *   Exploration of data structure (`df.info()`) and missing values (`df.isnull().sum()`).
    *   Data cleaning (handling missing values by filling).
*   **Task 2: Basic Data Analysis**
    *   Computation of basic statistics (`df.describe()`).
    *   Groupings on a categorical column ('species') and computation of mean for numerical columns.
    *   Identification of patterns and findings.
*   **Task 3: Data Visualization**
    *   Creation of at least four different types of visualizations:
        *   Line chart
        *   Bar chart
        *   Histogram
        *   Scatter plot
    *   Customization of plots with titles, axis labels, and legends.
*   **Error Handling:**
    *   Basic `try-except` blocks for file operations.

## Example Output (Console and Plots)

When run, the script will output text to the console detailing each step of the analysis, followed by displaying the plots.
