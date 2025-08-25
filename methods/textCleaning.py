import pandas as pd
import string
import re
import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "text_cleaning_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

def clean_text_columns(df, columns=None, operations=None, custom_patterns=None):
    """
    Performs comprehensive text cleaning on string columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of columns to clean (None for all text columns)
    - operations (list): List of operations to perform
    - custom_patterns (dict): Custom regex patterns to apply

    Returns:
    - dict: Result dictionary with status, data, and updates
    """
    if operations is None:
        operations = ['lowercase', 'remove_whitespace']
    
    status_updates = []
    
    try:
        df_cleaned = df.copy()
        
        # Identify text columns
        if columns is None:
            text_cols = df_cleaned.select_dtypes(include=['object', 'string']).columns.tolist()
        else:
            text_cols = [col for col in columns if col in df_cleaned.columns and df_cleaned[col].dtype in ['object', 'string']]
        
        if not text_cols:
            status_updates.append("No text columns found for cleaning")
            return {
                "status": "success",
                "message": "No text columns to clean",
                "updates": status_updates,
                "operations_used": operations,
                "columns_processed": [],
                "rows_before": len(df),
                "rows_after": len(df_cleaned),
                "columns_before": len(df.columns),
                "columns_after": len(df_cleaned.columns),
                "data": df_cleaned
            }
        
        status_updates.append(f"Processing {len(text_cols)} text columns: {', '.join(text_cols)}")
        logging.info(f"Starting text cleaning on {len(text_cols)} columns with operations: {operations}")
        
        for col in text_cols:
            original_col = df_cleaned[col].copy()
            changes_made = 0
            
            # Convert to string and handle NaN values
            df_cleaned[col] = df_cleaned[col].astype(str)
            df_cleaned[col] = df_cleaned[col].replace('nan', '')
            
            # Apply each operation
            for operation in operations:
                if operation == 'lowercase':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.lower()
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'uppercase':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.upper()
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_whitespace':
                    before = df_cleaned[col].copy()
                    # Remove leading/trailing whitespace and multiple spaces
                    df_cleaned[col] = df_cleaned[col].str.strip()
                    df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_punctuation':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.translate(str.maketrans('', '', string.punctuation))
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_numbers':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.replace(r'\d+', '', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_special_chars':
                    before = df_cleaned[col].copy()
                    # Keep only alphanumeric characters and spaces
                    df_cleaned[col] = df_cleaned[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_extra_spaces':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_html':
                    before = df_cleaned[col].copy()
                    df_cleaned[col] = df_cleaned[col].str.replace(r'<[^>]+>', '', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_urls':
                    before = df_cleaned[col].copy()
                    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
                    df_cleaned[col] = df_cleaned[col].str.replace(url_pattern, '', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                elif operation == 'remove_emails':
                    before = df_cleaned[col].copy()
                    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                    df_cleaned[col] = df_cleaned[col].str.replace(email_pattern, '', regex=True)
                    changes_made += (before != df_cleaned[col]).sum()
                    
                else:
                    status_updates.append(f"Warning: Unknown operation '{operation}' skipped")
            
            # Apply custom patterns if provided
            if custom_patterns:
                for pattern, replacement in custom_patterns.items():
                    try:
                        before = df_cleaned[col].copy()
                        df_cleaned[col] = df_cleaned[col].str.replace(pattern, replacement, regex=True)
                        pattern_changes = (before != df_cleaned[col]).sum()
                        changes_made += pattern_changes
                        if pattern_changes > 0:
                            status_updates.append(f"Column '{col}': Applied custom pattern '{pattern}' -> '{replacement}' ({pattern_changes} changes)")
                    except Exception as e:
                        status_updates.append(f"Error applying custom pattern '{pattern}' to column '{col}': {str(e)}")
            
            # Final cleanup - remove empty strings and convert back NaNs
            df_cleaned[col] = df_cleaned[col].replace('', pd.NA)
            
            if changes_made > 0:
                status_updates.append(f"Column '{col}': Made {changes_made} changes")
            else:
                status_updates.append(f"Column '{col}': No changes needed")
        
        # Additional post-processing
        for col in text_cols:
            try:
                # Remove rows where text columns became empty after cleaning (optional)
                original_nulls = original_col.isnull().sum()
                new_nulls = df_cleaned[col].isnull().sum()
                
                if new_nulls > original_nulls:
                    status_updates.append(f"Column '{col}': {new_nulls - original_nulls} additional empty values after cleaning")
            except:
                pass
        
        status_updates.append(f"Text cleaning completed successfully for {len(text_cols)} columns")
        logging.info(f"Text cleaning completed. Operations: {operations}")
        
        return {
            "status": "success",
            "operations_used": operations,
            "columns_processed": text_cols,
            "custom_patterns_applied": len(custom_patterns) if custom_patterns else 0,
            "rows_before": len(df),
            "rows_after": len(df_cleaned),
            "columns_before": len(df.columns),
            "columns_after": len(df_cleaned.columns),
            "updates": status_updates,
            "data": df_cleaned
        }
        
    except Exception as e:
        error_msg = f"Error in clean_text_columns: {e}"
        logging.error(error_msg)
        status_updates.append(f"Error: {error_msg}")
        return {
            "status": "error",
            "message": str(e),
            "updates": status_updates
        }