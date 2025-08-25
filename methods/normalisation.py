import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "normalisation_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

class Normalisation:
    def __init__(self):
        self.scalers = {}
        
    def normalize_data(self, file_path: str = None, df: pd.DataFrame = None, columns: list = None, method: str = "standard", 
                      feature_range: tuple = (0, 1), with_mean: bool = True, with_std: bool = True):
        """
        Normalizes numerical data using various methods.
        
        Args:
            file_path (str): Path to the CSV file (optional if df provided)
            df (pd.DataFrame): DataFrame to normalize (optional if file_path provided)
            columns (list): List of numerical columns to normalize (None for all numerical)
            method (str): Normalization method - "standard", "minmax", "robust", "normalize"
            feature_range (tuple): Range for min-max scaling (min, max)
            with_mean (bool): For standard scaling, center the data
            with_std (bool): For standard scaling, scale to unit variance
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
            logging.info(f"Before normalization, dataset has {len(data)} rows and {len(data.columns)} columns")

            # Identify numerical columns
            if columns is None:
                numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            else:
                # Filter to only include columns that exist and are numerical
                numerical_cols = []
                for col in columns:
                    if col in data.columns:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            numerical_cols.append(col)
                        else:
                            status_updates.append(f"Warning: Column '{col}' is not numeric")

            if not numerical_cols:
                status_updates.append("No numerical columns found for normalization")
                return {
                    "status": "success",
                    "message": "No numerical columns to normalize",
                    "updates": status_updates,
                    "method_used": method,
                    "columns_normalized": [],
                    "rows_before": int(len(data)),
                    "rows_after": int(len(data)),
                    "columns_before": int(len(data.columns)),
                    "columns_after": int(len(data.columns)),
                    "data": data
                }

            status_updates.append(f"Found {len(numerical_cols)} numerical columns: {', '.join(numerical_cols)}")
            logging.info(f"Normalizing {len(numerical_cols)} numerical columns")

            normalized_df = data.copy()
            
            # Handle missing values in numerical columns before normalization
            for col in numerical_cols:
                missing_count = normalized_df[col].isnull().sum()
                if missing_count > 0:
                    # Fill NaN with median for numerical columns
                    median_val = normalized_df[col].median()
                    if pd.isna(median_val):
                        median_val = 0  # Fallback if all values are NaN
                    normalized_df[col] = normalized_df[col].fillna(median_val)
                    status_updates.append(f"Filled {missing_count} missing values in {col} with median ({median_val:.3f})")

            # Apply normalization
            if method == "standard":
                status_updates.append("Applying standard scaling...")
                scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
                try:
                    scaled_data = scaler.fit_transform(normalized_df[numerical_cols])
                    normalized_df[numerical_cols] = scaled_data
                    self.scalers['standard'] = scaler
                    status_updates.append("Standard scaling completed")
                except Exception as e:
                    raise ValueError(f"Standard scaling failed: {str(e)}")

            elif method == "minmax":
                status_updates.append(f"Applying min-max scaling to range {feature_range}...")
                scaler = MinMaxScaler(feature_range=feature_range)
                try:
                    scaled_data = scaler.fit_transform(normalized_df[numerical_cols])
                    normalized_df[numerical_cols] = scaled_data
                    self.scalers['minmax'] = scaler
                    status_updates.append(f"Min-max scaling completed (range: {feature_range})")
                except Exception as e:
                    raise ValueError(f"Min-max scaling failed: {str(e)}")

            elif method == "robust":
                status_updates.append("Applying robust scaling...")
                scaler = RobustScaler()
                try:
                    scaled_data = scaler.fit_transform(normalized_df[numerical_cols])
                    normalized_df[numerical_cols] = scaled_data
                    self.scalers['robust'] = scaler
                    status_updates.append("Robust scaling completed")
                except Exception as e:
                    raise ValueError(f"Robust scaling failed: {str(e)}")

            elif method == "normalize":
                status_updates.append("Applying L2 normalization...")
                scaler = Normalizer(norm='l2')
                try:
                    # Normalizer works on rows, so we apply it to the numerical columns
                    normalized_values = scaler.fit_transform(normalized_df[numerical_cols])
                    normalized_df[numerical_cols] = normalized_values
                    self.scalers['normalize'] = scaler
                    status_updates.append("L2 normalization completed")
                except Exception as e:
                    raise ValueError(f"L2 normalization failed: {str(e)}")

            else:
                raise ValueError(f"Unknown normalization method: {method}")

            # Add statistics about the transformation
            for col in numerical_cols:
                try:
                    original_stats = data[col].describe()
                    normalized_stats = normalized_df[col].describe()
                    status_updates.append(f"Column {col}: mean {original_stats['mean']:.3f} -> {normalized_stats['mean']:.3f}, "
                                       f"std {original_stats['std']:.3f} -> {normalized_stats['std']:.3f}")
                except Exception as e:
                    status_updates.append(f"Could not compute stats for {col}: {str(e)}")

            # Handle infinite and NaN values after normalization
            for col in numerical_cols:
                # Replace infinite values
                inf_mask = np.isinf(normalized_df[col])
                if inf_mask.any():
                    inf_count = inf_mask.sum()
                    # Replace positive infinity with max finite value, negative infinity with min
                    finite_values = normalized_df[col][~inf_mask]
                    if len(finite_values) > 0:
                        max_finite = finite_values.max()
                        min_finite = finite_values.min()
                        normalized_df.loc[normalized_df[col] == np.inf, col] = max_finite
                        normalized_df.loc[normalized_df[col] == -np.inf, col] = min_finite
                    else:
                        normalized_df.loc[inf_mask, col] = 0
                    status_updates.append(f"Replaced {inf_count} infinite values in {col}")
                
                # Replace any remaining NaN values
                nan_mask = normalized_df[col].isnull()
                if nan_mask.any():
                    nan_count = nan_mask.sum()
                    normalized_df.loc[nan_mask, col] = 0
                    status_updates.append(f"Replaced {nan_count} NaN values in {col} with 0")

            # Final validation - ensure no problematic values remain
            for col in numerical_cols:
                problematic = normalized_df[col].isnull().sum() + np.isinf(normalized_df[col]).sum()
                if problematic > 0:
                    status_updates.append(f"Warning: {problematic} problematic values still remain in {col}")

            status_updates.append(f"Normalization completed. Final dataset has {len(normalized_df)} rows and {len(normalized_df.columns)} columns")
            logging.info(f"After normalization, dataset has {len(normalized_df)} rows and {len(normalized_df.columns)} columns")

            return {
                "status": "success",
                "method_used": method,
                "columns_normalized": numerical_cols,
                "rows_before": int(len(data)),
                "rows_after": int(len(normalized_df)),
                "columns_before": int(len(data.columns)),
                "columns_after": int(len(normalized_df.columns)),
                "updates": status_updates,
                "data": normalized_df
            }

        except Exception as e:
            error_msg = f"Error in normalize_data: {e}"
            logging.error(error_msg)
            status_updates.append(f"Error: {error_msg}")
            return {
                "status": "error",
                "message": str(e),
                "updates": status_updates
            }

    def inverse_transform(self, df: pd.DataFrame, columns: list = None, method: str = "standard"):
        """Reverse the normalization transformation."""
        try:
            if method not in self.scalers:
                raise ValueError(f"No scaler found for method: {method}")
            
            scaler = self.scalers[method]
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            result_df = df.copy()
            
            # Only inverse transform if scaler supports it (Normalizer doesn't have inverse_transform)
            if hasattr(scaler, 'inverse_transform') and method != 'normalize':
                result_df[columns] = scaler.inverse_transform(result_df[columns])
            else:
                logging.warning(f"Inverse transform not supported for method: {method}")
            
            return result_df
        except Exception as e:
            logging.error(f"Error in inverse_transform: {e}")
            return df

    def get_scaling_statistics(self, df: pd.DataFrame, columns: list = None):
        """Get statistics about the data before and after scaling."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        stats = {}
        for col in columns:
            if col in df.columns:
                try:
                    col_data = df[col].dropna()  # Remove NaN for statistics
                    
                    if len(col_data) == 0:
                        continue
                    
                    stats[col] = {
                        'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0,
                        'std': float(col_data.std()) if not pd.isna(col_data.std()) else 0.0,
                        'min': float(col_data.min()) if not pd.isna(col_data.min()) else 0.0,
                        'max': float(col_data.max()) if not pd.isna(col_data.max()) else 0.0,
                        'q25': float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else 0.0,
                        'q75': float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else 0.0,
                        'skewness': float(col_data.skew()) if not pd.isna(col_data.skew()) else 0.0,
                        'kurtosis': float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else 0.0
                    }
                except Exception as e:
                    logging.warning(f"Could not compute statistics for column {col}: {e}")
                    stats[col] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'q25': 0.0, 'q75': 0.0, 'skewness': 0.0, 'kurtosis': 0.0
                    }
        
        return stats

    def detect_outliers(self, df: pd.DataFrame, columns: list = None, method: str = "iqr", threshold: float = 1.5):
        """Detect outliers in numerical columns."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in columns:
            if col in df.columns:
                try:
                    col_data = df[col].dropna()
                    
                    if len(col_data) == 0:
                        outliers[col] = pd.Series(dtype='float64')
                        continue
                    
                    if method == "iqr":
                        Q1 = col_data.quantile(0.25)
                        Q3 = col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        
                        if IQR == 0:  # Handle case where IQR is 0
                            outliers[col] = pd.Series(dtype='float64')
                            continue
                            
                        lower_bound = Q1 - threshold * IQR
                        upper_bound = Q3 + threshold * IQR
                        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    
                    elif method == "zscore":
                        mean_val = col_data.mean()
                        std_val = col_data.std()
                        
                        if std_val == 0:  # Handle case where std is 0
                            outliers[col] = pd.Series(dtype='float64')
                            continue
                            
                        z_scores = np.abs((df[col] - mean_val) / std_val)
                        outliers[col] = df[z_scores > threshold][col]
                    
                    else:
                        raise ValueError(f"Unknown outlier detection method: {method}")
                        
                except Exception as e:
                    logging.warning(f"Could not detect outliers for column {col}: {e}")
                    outliers[col] = pd.Series(dtype='float64')
        
        return outliers