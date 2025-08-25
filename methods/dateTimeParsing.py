import pandas as pd

def parse_datetime_columns(df, date_format=None):
    """
    Converts object columns that look like dates into datetime objects.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - date_format (str, optional): The specific format of the date strings. 
                                   If None, pandas will try to infer it.

    Returns:
    - pd.DataFrame: The DataFrame with parsed datetime columns.
    """
    df_parsed = df.copy()
    for col in df_parsed.select_dtypes(include=['object', 'string']).columns:
        try:
            # Attempt to convert to datetime
            # errors='coerce' will turn unparseable values into NaT (Not a Time)
            converted_col = pd.to_datetime(df_parsed[col], format=date_format, errors='coerce')
            
            # Only replace the column if a significant portion could be converted
            if converted_col.notna().sum() > 0.5 * len(df_parsed[col]):
                df_parsed[col] = converted_col
        except Exception:
            continue
            
    return df_parsed