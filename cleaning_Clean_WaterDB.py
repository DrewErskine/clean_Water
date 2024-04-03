import pandas as pd

def load_dataset(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def inspect_dataset(df):
    """Manual inspection of the dataset."""
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns)
    print("Sample data:")
    print(df.head())

def clean_dataset(df):
    """Perform data cleaning."""
    initial_missing_count = df.isnull().sum().sum()
    df_cleaned = df.fillna(0)
    remaining_missing_count = df_cleaned.isnull().sum().sum()
    if remaining_missing_count == 0:
        print("No missing values remaining after data cleaning.")
    else:
        print("Warning: There are still some missing values remaining after data cleaning.")
    return df_cleaned


def save_selected_dataset(df_selected, output_file):
    """Save the cleaned and selected dataset."""
    df_selected.to_csv(output_file, index=False)
    print("Data cleaning and selection completed. Cleaned dataset saved as '{}'.".format(output_file))

def main():
    file_path = "water_potability.csv"
    output_file = "cleaned_selected_dataset.csv"
    
    # Load the dataset
    df = load_dataset(file_path)
    
    # Inspect the dataset
    inspect_dataset(df)
    
    # Perform data cleaning
    df_cleaned = clean_dataset(df)
    
    # Select important columns for the project
    important_columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
    df_selected = df_cleaned[important_columns]
    
    # Save the cleaned and selected dataset
    save_selected_dataset(df_selected, output_file)

if __name__ == "__main__":
    main()

