import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "encoding_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

class Encoding:
    def __init__(self):
        self.label_encoders = {}
        
    def encode_categorical_data(self, file_path: str = None, df: pd.DataFrame = None, 
                               columns: list = None, method: str = "label", 
                               drop_first: bool = False, handle_unknown: str = "error"):
        """
        Encodes categorical data using various methods.
        
        Args:
            file_path (str): Path to the CSV file (optional if df provided)
            df (pd.DataFrame): DataFrame to encode (optional if file_path provided)
            columns (list): List of columns to encode (None for all categorical)
            method (str): Encoding method - "label", "onehot", "target"
            drop_first (bool): For one-hot encoding, drop first category
            handle_unknown (str): How to handle unknown categories - "error", "ignore", "most_frequent"
        """
        status_updates = []

        try:
            # Handle input data - either DataFrame or file path
            if df is not None:
                data = df.copy()
                status_updates.append("Using provided DataFrame")
            elif file_path is not None:
                status_updates.append("Loading dataset from file...")
                data = pd.read_csv(file_path)
            else:
                raise ValueError("Either file_path or df must be provided")
                
            status_updates.append(f"Dataset loaded with {len(data)} rows and {len(data.columns)} columns")
            logging.info(f"Before encoding, dataset has {len(data)} rows and {len(data.columns)} columns")

            # Identify categorical columns (object and category dtypes)
            if columns is None:
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                # Filter to only include columns that exist and are categorical
                categorical_cols = []
                for col in columns:
                    if col not in data.columns:
                        status_updates.append(f"Warning: Column '{col}' not found in dataset")
                        continue
                    if data[col].dtype not in ['object', 'category']:
                        status_updates.append(f"Warning: Column '{col}' is not categorical (dtype: {data[col].dtype})")
                        continue
                    categorical_cols.append(col)

            if not categorical_cols:
                status_updates.append("No categorical columns found for encoding")
                return {
                    "status": "success",
                    "message": "No categorical columns to encode",
                    "updates": status_updates,
                    "method_used": method,
                    "columns_encoded": [],
                    "rows_before": len(data),
                    "rows_after": len(data),
                    "columns_before": len(data.columns),
                    "columns_after": len(data.columns),
                    "data": data
                }

            status_updates.append(f"Found {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
            logging.info(f"Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

            encoded_df = data.copy()

            if method == "label":
                status_updates.append("Applying label encoding...")
                successfully_encoded = []
                
                for col in categorical_cols:
                    try:
                        # Handle missing values by filling with a placeholder
                        original_nulls = encoded_df[col].isnull().sum()
                        if original_nulls > 0:
                            encoded_df[col] = encoded_df[col].fillna('__MISSING__')
                            status_updates.append(f"Column '{col}': Filled {original_nulls} missing values with placeholder")
                        
                        # Convert to string to handle mixed types
                        encoded_df[col] = encoded_df[col].astype(str)
                        
                        # Apply label encoding
                        le = LabelEncoder()
                        encoded_df[col] = le.fit_transform(encoded_df[col])
                        self.label_encoders[col] = le
                        
                        successfully_encoded.append(col)
                        unique_values = len(le.classes_)
                        status_updates.append(f"Label encoded column '{col}': {unique_values} unique values -> integers 0-{unique_values-1}")
                        
                    except Exception as e:
                        error_msg = f"Failed to encode column '{col}': {str(e)}"
                        status_updates.append(error_msg)
                        logging.warning(error_msg)
                        continue
                
                if successfully_encoded:
                    status_updates.append(f"Label encoding completed successfully for {len(successfully_encoded)} columns")
                else:
                    raise Exception("No columns were successfully encoded")

            elif method == "onehot":
                status_updates.append("Applying one-hot encoding...")
                columns_to_encode = []
                skipped_columns = []
                
                for col in categorical_cols:
                    # Check number of unique values
                    unique_count = encoded_df[col].nunique()
                    if unique_count > 50:  # Too many categories for one-hot
                        skipped_columns.append(f"{col} ({unique_count} categories)")
                        status_updates.append(f"Column '{col}' has too many categories ({unique_count}), skipping")
                        continue
                    columns_to_encode.append(col)
                
                if skipped_columns:
                    status_updates.append(f"Skipped columns with too many categories: {', '.join(skipped_columns)}")
                
                if columns_to_encode:
                    # Use pandas get_dummies for one-hot encoding
                    try:
                        dummies = pd.get_dummies(
                            encoded_df[columns_to_encode], 
                            prefix=columns_to_encode, 
                            drop_first=drop_first,
                            dummy_na=True  # Handle NaN values
                        )
                        
                        # Remove original categorical columns and add dummy columns
                        encoded_df = encoded_df.drop(columns=columns_to_encode)
                        encoded_df = pd.concat([encoded_df, dummies], axis=1)
                        
                        status_updates.append(f"One-hot encoded {len(columns_to_encode)} columns -> {len(dummies.columns)} new columns")
                        status_updates.append(f"New columns: {', '.join(dummies.columns[:5])}{'...' if len(dummies.columns) > 5 else ''}")
                        
                    except Exception as e:
                        raise Exception(f"One-hot encoding failed: {str(e)}")
                else:
                    status_updates.append("No columns suitable for one-hot encoding")
                
                status_updates.append("One-hot encoding completed successfully")

            elif method == "target":
                status_updates.append("Applying target encoding (using frequency encoding)...")
                successfully_encoded = []
                
                # Since we don't have a target variable, use frequency encoding
                for col in categorical_cols:
                    try:
                        # Calculate frequency encoding (proportion of each category)
                        value_counts = encoded_df[col].value_counts(normalize=True, dropna=False)
                        
                        # Create new column with frequency values
                        new_col_name = f"{col}_freq_encoded"
                        encoded_df[new_col_name] = encoded_df[col].map(value_counts)
                        
                        # Fill any missing mappings with 0 (shouldn't happen with dropna=False)
                        encoded_df[new_col_name] = encoded_df[new_col_name].fillna(0)
                        
                        successfully_encoded.append(col)
                        status_updates.append(f"Frequency encoded column '{col}' -> '{new_col_name}'")
                        
                    except Exception as e:
                        error_msg = f"Failed to frequency encode column '{col}': {str(e)}"
                        status_updates.append(error_msg)
                        logging.warning(error_msg)
                        continue
                
                if successfully_encoded:
                    status_updates.append(f"Target encoding completed successfully for {len(successfully_encoded)} columns")
                else:
                    raise Exception("No columns were successfully target encoded")

            else:
                raise ValueError(f"Unknown encoding method: {method}. Supported methods: 'label', 'onehot', 'target'")

            # Final validation
            if encoded_df.empty:
                raise Exception("Encoding resulted in empty dataset")
            
            # Check for any issues
            inf_count = np.isinf(encoded_df.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                status_updates.append(f"Warning: {inf_count} infinite values detected after encoding")
            
            status_updates.append(f"Encoding completed. Final dataset has {len(encoded_df)} rows and {len(encoded_df.columns)} columns")
            logging.info(f"After encoding, dataset has {len(encoded_df)} rows and {len(encoded_df.columns)} columns")

            return {
                "status": "success",
                "method_used": method,
                "columns_encoded": categorical_cols,
                "rows_before": len(data),
                "rows_after": len(encoded_df),
                "columns_before": len(data.columns),
                "columns_after": len(encoded_df.columns),
                "updates": status_updates,
                "data": encoded_df
            }

        except Exception as e:
            error_msg = f"Error in encode_categorical_data: {str(e)}"
            logging.error(error_msg)
            status_updates.append(f"Error: {error_msg}")
            return {
                "status": "error",
                "message": str(e),
                "updates": status_updates
            }

    def get_encoding_mapping(self, column: str):
        """Get the mapping for a specific column's label encoding."""
        if column in self.label_encoders:
            le = self.label_encoders[column]
            try:
                return dict(zip(le.classes_, le.transform(le.classes_)))
            except Exception as e:
                logging.warning(f"Could not get encoding mapping for column {column}: {e}")
                return None
        return None

    def inverse_transform(self, df: pd.DataFrame, columns: list = None):
        """Reverse the label encoding transformation."""
        if columns is None:
            columns = list(self.label_encoders.keys())
        
        result_df = df.copy()
        for col in columns:
            if col in self.label_encoders and col in result_df.columns:
                le = self.label_encoders[col]
                try:
                    # Only inverse transform if column contains valid encoded values
                    max_encoded_value = len(le.classes_) - 1
                    valid_mask = (result_df[col] >= 0) & (result_df[col] <= max_encoded_value)
                    
                    if valid_mask.any():
                        result_df.loc[valid_mask, col] = le.inverse_transform(
                            result_df.loc[valid_mask, col].astype(int)
                        )
                    
                except Exception as e:
                    logging.warning(f"Could not inverse transform column {col}: {e}")
        
        return result_df

    def get_categorical_summary(self, df: pd.DataFrame):
        """Get summary statistics for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        summary = {}
        
        for col in categorical_cols:
            try:
                col_data = df[col].dropna()
                summary[col] = {
                    'unique_count': int(col_data.nunique()),
                    'missing_count': int(df[col].isnull().sum()),
                    'most_frequent': str(col_data.mode().iloc[0]) if not col_data.empty else 'N/A',
                    'data_type': str(df[col].dtype),
                    'memory_usage_kb': float(df[col].memory_usage(deep=True) / 1024)
                }
            except Exception as e:
                logging.warning(f"Could not get summary for column {col}: {e}")
                summary[col] = {'error': str(e)}
        
        return summary