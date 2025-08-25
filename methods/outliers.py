import pandas as pd
import numpy as np
import logging
import os
from sklearn.ensemble import IsolationForest

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "outliers_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

def handle_outliers(df, columns=None, method='iqr', action='remove', threshold=3):
    """
    Handles outliers in the numerical columns of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of columns to process (None for all numeric columns)
    - method (str): The method to use for outlier detection ('iqr', 'zscore', 'modified_zscore', 'isolation_forest').
    - action (str): Action to take ('remove', 'cap', 'transform')
    - threshold (float): The threshold for outlier detection. 
                         For IQR, it's the multiplier. 
                         For Z-score, it's the number of standard deviations.

    Returns:
    - dict: Result dictionary with status, data, and updates
    """
    status_updates = []
    
    try:
        df_cleaned = df.copy()
        
        # Identify numerical columns
        if columns is None:
            numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numerical_cols = [col for col in columns if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col])]
        
        if not numerical_cols:
            status_updates.append("No numerical columns found for outlier detection")
            return {
                "status": "success",
                "message": "No numerical columns to process",
                "updates": status_updates,
                "method_used": method,
                "action_used": action,
                "columns_processed": [],
                "outliers_detected": 0,
                "rows_before": len(df),
                "rows_after": len(df_cleaned),
                "columns_before": len(df.columns),
                "columns_after": len(df_cleaned.columns),
                "data": df_cleaned
            }
        
        status_updates.append(f"Processing {len(numerical_cols)} numerical columns: {', '.join(numerical_cols)}")
        logging.info(f"Processing outliers in {len(numerical_cols)} columns using {method} method")
        
        total_outliers = 0
        outlier_indices = set()
        
        for col in numerical_cols:
            col_data = df_cleaned[col].dropna()
            if len(col_data) == 0:
                continue
                
            outlier_mask = pd.Series(False, index=df_cleaned.index)
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                if IQR == 0:
                    status_updates.append(f"Column {col}: IQR is 0, skipping outlier detection")
                    continue
                    
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                
            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                
                if std == 0:
                    status_updates.append(f"Column {col}: Standard deviation is 0, skipping outlier detection")
                    continue
                    
                z_scores = np.abs((df_cleaned[col] - mean) / std)
                outlier_mask = z_scores > threshold
                
            elif method == 'modified_zscore':
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                
                if mad == 0:
                    status_updates.append(f"Column {col}: MAD is 0, skipping outlier detection")
                    continue
                    
                modified_z_scores = 0.6745 * (df_cleaned[col] - median) / mad
                outlier_mask = np.abs(modified_z_scores) > threshold
                
            elif method == 'isolation_forest':
                try:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(col_data.values.reshape(-1, 1))
                    outlier_mask = pd.Series(outlier_pred == -1, index=col_data.index)
                    # Reindex to match the full dataframe
                    outlier_mask = outlier_mask.reindex(df_cleaned.index, fill_value=False)
                except Exception as e:
                    status_updates.append(f"Column {col}: Isolation Forest failed: {str(e)}")
                    continue
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            # Count outliers for this column
            col_outliers = outlier_mask.sum()
            total_outliers += col_outliers
            
            if col_outliers > 0:
                status_updates.append(f"Column {col}: Found {col_outliers} outliers")
                
                if action == 'remove':
                    outlier_indices.update(outlier_mask[outlier_mask].index)
                    
                elif action == 'cap':
                    if method in ['iqr', 'zscore', 'modified_zscore']:
                        if method == 'iqr':
                            lower_bound = Q1 - threshold * IQR
                            upper_bound = Q3 + threshold * IQR
                        elif method == 'zscore':
                            lower_bound = mean - threshold * std
                            upper_bound = mean + threshold * std
                        elif method == 'modified_zscore':
                            lower_bound = median - threshold * mad / 0.6745
                            upper_bound = median + threshold * mad / 0.6745
                        
                        df_cleaned.loc[df_cleaned[col] < lower_bound, col] = lower_bound
                        df_cleaned.loc[df_cleaned[col] > upper_bound, col] = upper_bound
                        
                elif action == 'transform':
                    # Apply log transformation to positive values
                    positive_mask = df_cleaned[col] > 0
                    if positive_mask.any():
                        df_cleaned.loc[positive_mask, col] = np.log1p(df_cleaned.loc[positive_mask, col])
                        status_updates.append(f"Column {col}: Applied log transformation")
        
        # Remove outlier rows if action is 'remove'
        if action == 'remove' and outlier_indices:
            df_cleaned = df_cleaned.drop(outlier_indices).reset_index(drop=True)
            status_updates.append(f"Removed {len(outlier_indices)} outlier rows")
        
        status_updates.append(f"Outlier handling completed. Total outliers detected: {total_outliers}")
        logging.info(f"Outlier handling completed. Method: {method}, Action: {action}, Total outliers: {total_outliers}")
        
        return {
            "status": "success",
            "method_used": method,
            "action_used": action,
            "columns_processed": numerical_cols,
            "outliers_detected": int(total_outliers),
            "rows_before": len(df),
            "rows_after": len(df_cleaned),
            "columns_before": len(df.columns),
            "columns_after": len(df_cleaned.columns),
            "updates": status_updates,
            "data": df_cleaned
        }
        
    except Exception as e:
        error_msg = f"Error in handle_outliers: {e}"
        logging.error(error_msg)
        status_updates.append(f"Error: {error_msg}")
        return {
            "status": "error",
            "message": str(e),
            "updates": status_updates
        }