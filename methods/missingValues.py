import pandas as pd
import logging
import numpy as np

logging.basicConfig(
    filename="missing_values_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MissingValues:
    def fix_missing_values(self, file_path: str = None, df: pd.DataFrame = None, strategy: str = "drop_rows", threshold: float = 0.5):
        """
        Handles missing values and returns result + status updates.
        
        Args:
            file_path (str): Path to the CSV file (optional if df provided)
            df (pd.DataFrame): DataFrame to process (optional if file_path provided)
            strategy (str): Strategy to handle missing values
                - "drop_rows": Remove rows with any missing values
                - "drop_rows_threshold": Remove rows with missing values above threshold
                - "drop_columns": Remove columns with any missing values  
                - "drop_columns_threshold": Remove columns with missing values above threshold
                - "fill_mean": Fill numeric columns with mean, categorical with mode
                - "fill_median": Fill numeric columns with median, categorical with mode
                - "fill_mode": Fill all columns with mode (most frequent value)
                - "forward_fill": Forward fill missing values
                - "backward_fill": Backward fill missing values
            threshold (float): Threshold for drop strategies (0.0 to 1.0)
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
            logging.info(f"Before cleaning, dataset has {len(data)} rows and {len(data.columns)} columns")

            # Calculate missing values info
            total_missing = data.isnull().sum().sum()
            missing_by_column = data.isnull().sum()
            missing_columns = missing_by_column[missing_by_column > 0]
            
            status_updates.append(f"Found {total_missing} total missing values across {len(missing_columns)} columns")
            logging.info(f"Found {total_missing} total missing values")

            if total_missing == 0:
                status_updates.append("No missing values found - no cleaning needed")
                return {
                    "status": "success",
                    "total_missing_before": 0,
                    "total_missing_after": 0,
                    "rows_before": len(data),
                    "rows_after": len(data),
                    "columns_before": len(data.columns),
                    "columns_after": len(data.columns),
                    "strategy_used": strategy,
                    "updates": status_updates,
                    "data": data
                }

            # Create a copy for processing
            processed_data = data.copy()

            # Apply selected strategy
            if strategy == "drop_rows":
                processed_data = processed_data.dropna()
                status_updates.append("Dropped all rows with missing values")

            elif strategy == "drop_rows_threshold":
                min_count = int(threshold * len(processed_data.columns))
                processed_data = processed_data.dropna(thresh=min_count)
                status_updates.append(f"Dropped rows with more than {(1-threshold)*100:.1f}% missing values")

            elif strategy == "drop_columns":
                processed_data = processed_data.dropna(axis=1)
                status_updates.append("Dropped all columns with missing values")

            elif strategy == "drop_columns_threshold":
                min_count = int(threshold * len(processed_data))
                processed_data = processed_data.dropna(axis=1, thresh=min_count)
                status_updates.append(f"Dropped columns with more than {(1-threshold)*100:.1f}% missing values")

            elif strategy == "fill_mean":
                numeric_cols = processed_data.select_dtypes(include=['number']).columns
                categorical_cols = processed_data.select_dtypes(exclude=['number']).columns
                
                # Fill numeric columns with mean
                for col in numeric_cols:
                    if processed_data[col].isnull().any():
                        mean_val = processed_data[col].mean()
                        # Check if mean is valid (not NaN or infinite)
                        if pd.isna(mean_val) or np.isinf(mean_val):
                            # Use median as fallback, or 0 if both fail
                            fallback_val = processed_data[col].median()
                            if pd.isna(fallback_val) or np.isinf(fallback_val):
                                fallback_val = 0
                            processed_data[col].fillna(fallback_val, inplace=True)
                            status_updates.append(f"Column '{col}': Used fallback value {fallback_val} (mean was invalid)")
                        else:
                            processed_data[col].fillna(mean_val, inplace=True)
                            status_updates.append(f"Column '{col}': Filled with mean {mean_val:.3f}")
                
                # Fill categorical columns with mode
                for col in categorical_cols:
                    if processed_data[col].isnull().any():
                        mode_series = processed_data[col].mode()
                        mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                        processed_data[col].fillna(mode_val, inplace=True)
                        status_updates.append(f"Column '{col}': Filled with mode '{mode_val}'")
                
                status_updates.append("Filled numeric columns with mean, categorical with mode")

            elif strategy == "fill_median":
                numeric_cols = processed_data.select_dtypes(include=['number']).columns
                categorical_cols = processed_data.select_dtypes(exclude=['number']).columns
                
                # Fill numeric columns with median
                for col in numeric_cols:
                    if processed_data[col].isnull().any():
                        median_val = processed_data[col].median()
                        # Check if median is valid (not NaN or infinite)
                        if pd.isna(median_val) or np.isinf(median_val):
                            # Use 0 as fallback
                            processed_data[col].fillna(0, inplace=True)
                            status_updates.append(f"Column '{col}': Used fallback value 0 (median was invalid)")
                        else:
                            processed_data[col].fillna(median_val, inplace=True)
                            status_updates.append(f"Column '{col}': Filled with median {median_val:.3f}")
                
                # Fill categorical columns with mode
                for col in categorical_cols:
                    if processed_data[col].isnull().any():
                        mode_series = processed_data[col].mode()
                        mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                        processed_data[col].fillna(mode_val, inplace=True)
                        status_updates.append(f"Column '{col}': Filled with mode '{mode_val}'")
                
                status_updates.append("Filled numeric columns with median, categorical with mode")

            elif strategy == "fill_mode":
                for col in processed_data.columns:
                    if processed_data[col].isnull().any():
                        mode_series = processed_data[col].mode()
                        mode_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                        processed_data[col].fillna(mode_value, inplace=True)
                        status_updates.append(f"Column '{col}': Filled with mode '{mode_value}'")
                
                status_updates.append("Filled all columns with mode (most frequent value)")

            elif strategy == "forward_fill":
                processed_data = processed_data.ffill()
                status_updates.append("Applied forward fill to missing values")

            elif strategy == "backward_fill":
                processed_data = processed_data.bfill()
                status_updates.append("Applied backward fill to missing values")

            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Calculate final missing values
            total_missing_after = processed_data.isnull().sum().sum()
            
            status_updates.append(f"After cleaning: {len(processed_data)} rows, {len(processed_data.columns)} columns")
            status_updates.append(f"Missing values reduced from {total_missing} to {total_missing_after}")
            
            logging.info(f"After cleaning: {len(processed_data)} rows, {len(processed_data.columns)} columns")
            logging.info(f"Missing values: {total_missing} -> {total_missing_after}")

            return {
                "status": "success",
                "total_missing_before": int(total_missing),
                "total_missing_after": int(total_missing_after),
                "rows_before": len(data),
                "rows_after": len(processed_data),
                "columns_before": len(data.columns),
                "columns_after": len(processed_data.columns),
                "strategy_used": strategy,
                "missing_by_column": missing_by_column.to_dict(),
                "updates": status_updates,
                "data": processed_data
            }

        except Exception as e:
            logging.error(f"Error in fix_missing_values: {e}")
            return {
                "status": "error",
                "message": str(e),
                "updates": status_updates
            }