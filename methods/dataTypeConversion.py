import pandas as pd
import numpy as np
import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "data_type_conversion_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

def convert_data_types(df, type_mapping=None, auto_detect=True, errors='coerce'):
    """
    Automatically converts columns of a DataFrame to the best possible data types.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - type_mapping (dict): Manual mapping of column names to desired types
    - auto_detect (bool): Whether to automatically detect and convert types
    - errors (str): How to handle conversion errors ('coerce', 'raise', 'ignore')

    Returns:
    - dict: Result dictionary with status, data, and updates
    """
    status_updates = []
    
    try:
        df_converted = df.copy()
        conversions_made = {}
        
        status_updates.append(f"Starting data type conversion for {len(df_converted.columns)} columns")
        logging.info(f"Starting data type conversion. Auto-detect: {auto_detect}")
        
        # Apply manual type mapping first if provided
        if type_mapping:
            status_updates.append(f"Applying manual type mappings for {len(type_mapping)} columns")
            
            for col, target_type in type_mapping.items():
                if col not in df_converted.columns:
                    status_updates.append(f"Warning: Column '{col}' not found in dataset")
                    continue
                    
                original_type = str(df_converted[col].dtype)
                
                try:
                    if target_type in ['int', 'int64', 'int32']:
                        df_converted[col] = pd.to_numeric(df_converted[col], errors=errors).astype('int64')
                    elif target_type in ['float', 'float64', 'float32']:
                        df_converted[col] = pd.to_numeric(df_converted[col], errors=errors)
                    elif target_type in ['str', 'string', 'object']:
                        df_converted[col] = df_converted[col].astype(str)
                    elif target_type in ['datetime', 'datetime64']:
                        df_converted[col] = pd.to_datetime(df_converted[col], errors=errors)
                    elif target_type in ['bool', 'boolean']:
                        df_converted[col] = df_converted[col].astype(bool)
                    elif target_type in ['category']:
                        df_converted[col] = df_converted[col].astype('category')
                    else:
                        status_updates.append(f"Warning: Unknown target type '{target_type}' for column '{col}'")
                        continue
                    
                    new_type = str(df_converted[col].dtype)
                    conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'manual'}
                    status_updates.append(f"Column '{col}': {original_type} -> {new_type} (manual)")
                    
                except Exception as e:
                    status_updates.append(f"Failed to convert column '{col}' to {target_type}: {str(e)}")
                    continue
        
        # Auto-detect and convert types if enabled
        if auto_detect:
            status_updates.append("Starting automatic type detection...")
            
            for col in df_converted.columns:
                # Skip if already converted manually
                if col in (type_mapping or {}):
                    continue
                    
                original_type = str(df_converted[col].dtype)
                original_col = df_converted[col].copy()
                
                # Try to convert to numeric first
                try:
                    numeric_converted = pd.to_numeric(df_converted[col], errors='coerce')
                    non_null_numeric = numeric_converted.dropna()
                    
                    # If a significant portion can be converted to numeric
                    if len(non_null_numeric) > 0.7 * len(df_converted[col].dropna()):
                        # Check if it should be integer or float
                        if len(non_null_numeric) > 0 and all(x.is_integer() for x in non_null_numeric if pd.notna(x)):
                            try:
                                df_converted[col] = numeric_converted.astype('Int64')  # Nullable integer
                                new_type = str(df_converted[col].dtype)
                                conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_int'}
                                status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected integer)")
                                continue
                            except:
                                df_converted[col] = numeric_converted
                                new_type = str(df_converted[col].dtype)
                                conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_float'}
                                status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected float)")
                                continue
                        else:
                            df_converted[col] = numeric_converted
                            new_type = str(df_converted[col].dtype)
                            conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_float'}
                            status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected float)")
                            continue
                except:
                    pass
                
                # Try to convert to datetime
                if original_type == 'object':
                    try:
                        datetime_converted = pd.to_datetime(df_converted[col], errors='coerce', infer_datetime_format=True)
                        non_null_datetime = datetime_converted.dropna()
                        
                        # If a significant portion can be converted to datetime
                        if len(non_null_datetime) > 0.5 * len(df_converted[col].dropna()):
                            df_converted[col] = datetime_converted
                            new_type = str(df_converted[col].dtype)
                            conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_datetime'}
                            status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected datetime)")
                            continue
                    except Exception as e:
                        pass
                
                # Check for boolean values
                if original_type == 'object':
                    unique_values = set(str(x).lower() for x in df_converted[col].dropna().unique())
                    boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n', 't', 'f'}
                    
                    if unique_values.issubset(boolean_values) and len(unique_values) > 1:
                        try:
                            # Map common boolean representations
                            boolean_map = {
                                'true': True, 'false': False, '1': True, '0': False,
                                'yes': True, 'no': False, 'y': True, 'n': False,
                                't': True, 'f': False
                            }
                            
                            df_converted[col] = df_converted[col].astype(str).str.lower().map(boolean_map)
                            new_type = str(df_converted[col].dtype)
                            conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_boolean'}
                            status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected boolean)")
                            continue
                        except:
                            pass
                
                # Check if categorical conversion would be beneficial
                if original_type == 'object':
                    unique_count = df_converted[col].nunique()
                    total_count = len(df_converted[col])
                    
                    # If less than 50% unique values and not too many categories
                    if unique_count < total_count * 0.5 and unique_count < 100:
                        try:
                            df_converted[col] = df_converted[col].astype('category')
                            new_type = str(df_converted[col].dtype)
                            conversions_made[col] = {'from': original_type, 'to': new_type, 'method': 'auto_category'}
                            status_updates.append(f"Column '{col}': {original_type} -> {new_type} (auto-detected category)")
                            continue
                        except:
                            pass
        
        # Final cleanup - handle any remaining issues
        for col in df_converted.columns:
            if df_converted[col].dtype == 'object':
                # Ensure string columns don't have mixed types
                try:
                    df_converted[col] = df_converted[col].astype(str)
                except:
                    pass
        
        # Calculate memory usage improvement
        try:
            original_memory = df.memory_usage(deep=True).sum()
            new_memory = df_converted.memory_usage(deep=True).sum()
            memory_saved = original_memory - new_memory
            memory_saved_mb = memory_saved / (1024 * 1024)
            
            status_updates.append(f"Memory usage: {original_memory / (1024 * 1024):.2f} MB -> {new_memory / (1024 * 1024):.2f} MB")
            if memory_saved > 0:
                status_updates.append(f"Memory saved: {memory_saved_mb:.2f} MB ({(memory_saved/original_memory)*100:.1f}%)")
        except:
            pass
        
        status_updates.append(f"Data type conversion completed. {len(conversions_made)} columns converted.")
        logging.info(f"Data type conversion completed. Conversions: {len(conversions_made)}")
        
        return {
            "status": "success",
            "conversions_made": conversions_made,
            "auto_detect_used": auto_detect,
            "rows_before": len(df),
            "rows_after": len(df_converted),
            "columns_before": len(df.columns),
            "columns_after": len(df_converted.columns),
            "updates": status_updates,
            "data": df_converted
        }
        
    except Exception as e:
        error_msg = f"Error in convert_data_types: {e}"
        logging.error(error_msg)
        status_updates.append(f"Error: {error_msg}")
        return {
            "status": "error",
            "message": str(e),
            "updates": status_updates
        }   