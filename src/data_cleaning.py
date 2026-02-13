import pandas as pd

def load_data(path):
    """
    Load dataset from given path
    """
    df = pd.read_csv(path)
    return df

def clean_data(df):
    """
    Perform data cleaning:
    - Handle missing values
    - Drop unnecessary columns
    """
    
    num_cols = ["Bathroom", "Parking"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    cat_cols = ["Furnishing", "Type"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    if "Per_Sqft" in df.columns:
        df = df.drop("Per_Sqft", axis=1)

    return df
