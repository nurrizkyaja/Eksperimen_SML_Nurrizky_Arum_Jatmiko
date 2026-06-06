import os
import pandas as pd
from pathlib import Path


def extract_ticket_number(ticket: str) -> str:
    """Extracts the ticket number from the ticket string."""
    return ticket.split(" ")[-1]


def extract_ticket_item(ticket: str) -> str:
    """Extracts the ticket item/prefix from the ticket string."""
    items = ticket.split(" ")
    if len(items) == 1:
        return "NONE"
    return "_".join(items[:-1])


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and preprocesses the Titanic dataset.

    Args:
        df (pd.DataFrame): Raw DataFrame containing Titanic data.

    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Drop unnecessary columns
    columns_to_drop = ["Name", "Cabin", "PassengerId"]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Impute missing values for Age with median
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())

    # Process Ticket column
    if 'Ticket' in df.columns:
        df["Ticket_number"] = df["Ticket"].apply(extract_ticket_number)
        df["Ticket_item"] = df["Ticket"].apply(extract_ticket_item)
        df = df.drop(columns=["Ticket"], errors='ignore')

    return df


def main() -> None:
    """Main execution function for automated preprocessing."""
    # Define file paths using pathlib for cross-platform compatibility
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "titanic_raw"
    preprocessed_dir = base_dir / "preprocessing"

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    # Verify raw data exists
    if not train_path.exists() or not test_path.exists():
        print(f"Error: Raw data files not found in {raw_dir}.")
        return

    # Load data
    print("Loading raw data...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Preprocess data
    print("Preprocessing data...")
    processed_train_df = preprocess_data(df_train)
    processed_test_df = preprocess_data(df_test)

    # Save processed data
    output_dir = preprocessed_dir / "titanic_preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_path = output_dir / "titanic_preprocessed_train.csv"
    test_output_path = output_dir / "titanic_preprocessed_test.csv"
    processed_train_df.to_csv(train_output_path, index=False)
    processed_test_df.to_csv(test_output_path, index=False)

    print(f"Preprocessing complete. Files saved to:")
    print(f" - {train_output_path}")
    print(f" - {test_output_path}")


if __name__ == "__main__":
    main()
