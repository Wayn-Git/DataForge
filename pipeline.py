import pandas as pd
import numpy as np
import os
import json
import logging
import sys
import math
from datetime import datetime

# Import the methods - create a methods directory structure
try:
    from methods.missingValues import MissingValues
    from methods.duplicate import Duplicate
    from methods.encoding import Encoding
    from methods.spellingFix import TypoFix
    from methods.normalisation import Normalisation
    from methods.outliers import handle_outliers
    from methods.dataTypeConversion import convert_data_types
    from methods.textCleaning import clean_text_columns
    from methods.dateTimeParsing import parse_datetime_columns
except ImportError:
    # Fallback - try importing from current directory
    try:
        from missingValues import MissingValues
        from duplicate import Duplicate
        from encoding import Encoding
        from spellingFix import TypoFix
        from normalisation import Normalisation
        from outliers import handle_outliers
        from dataTypeConversion import convert_data_types
        from textCleaning import clean_text_columns
        from dateTimeParsing import parse_datetime_columns
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all method files are in the correct directory structure")
        sys.exit(1)

# Configure logging
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(current_dir, "pipeline_log.txt")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

def sanitize_for_json(obj):
    """
    Recursively sanitize data to make it JSON-compliant.
    Handles NaN, infinity, and other problematic values.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None  # Convert NaN/inf to null
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'item'):  # numpy scalars
        return sanitize_for_json(obj.item())
    else:
        return obj

def clean_dataframe_for_processing(df):
    """
    Clean a DataFrame to ensure it can be processed and saved properly.
    """
    cleaned_df = df.copy()
    
    # Replace inf and -inf with NaN, then handle appropriately
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Replace infinite values with NaN
        cleaned_df[col] = cleaned_df[col].replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with column median, or 0 if all NaN
        if cleaned_df[col].isnull().all():
            cleaned_df[col] = 0
        else:
            median_val = cleaned_df[col].median()
            if pd.isna(median_val):
                cleaned_df[col] = cleaned_df[col].fillna(0)
            else:
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
    
    # Handle non-numeric columns
    non_numeric_cols = cleaned_df.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        cleaned_df[col] = cleaned_df[col].fillna('')
    
    return cleaned_df

class DataCleaningPipeline:
    def __init__(self):
        self.missing_values_handler = MissingValues()
        self.duplicate_handler = Duplicate()
        self.encoding_handler = Encoding()
        self.typo_handler = TypoFix()
        self.normalization_handler = Normalisation()
        self.current_file_path = None
        self.cleaning_history = []
        
    def run_pipeline(self, file_path: str, operations: dict, output_path: str = None):
        """
        Run the complete data cleaning pipeline.
        """
        self.current_file_path = file_path
        pipeline_results = {
            "status": "success",
            "start_time": datetime.now().isoformat(),
            "operations": [],
            "final_data_shape": None,
            "total_processing_time": None,
            "errors": []
        }
        
        try:
            logging.info(f"Starting pipeline for file: {file_path}")
            
            # Load initial data once
            try:
                current_df = pd.read_csv(file_path)
                current_df = clean_dataframe_for_processing(current_df)
                logging.info(f"Loaded initial dataset: {current_df.shape}")
            except Exception as e:
                error_msg = f"Failed to load input file: {str(e)}"
                logging.error(error_msg)
                pipeline_results["errors"].append(error_msg)
                pipeline_results["status"] = "error"
                return pipeline_results
            
            # Define operation order (important for data cleaning)
            operation_order = [
                "data_type_conversion",
                "text_cleaning", 
                "datetime_parsing",
                "missing_values",
                "duplicates",
                "outliers",
                "typo_fix",
                "encoding",
                "normalization"
            ]
            
            # Execute operations in order
            for operation in operation_order:
                if not operations.get(operation, {}).get("enabled", False):
                    continue
                
                logging.info(f"Running {operation} operation...")
                
                try:
                    if operation == "missing_values":
                        result = self._handle_missing_values(current_df, operations[operation])
                    elif operation == "duplicates":
                        result = self._handle_duplicates(current_df, operations[operation])
                    elif operation == "outliers":
                        result = self._handle_outliers(current_df, operations[operation])
                    elif operation == "data_type_conversion":
                        result = self._handle_data_type_conversion(current_df, operations[operation])
                    elif operation == "text_cleaning":
                        result = self._handle_text_cleaning(current_df, operations[operation])
                    elif operation == "datetime_parsing":
                        result = self._handle_datetime_parsing(current_df, operations[operation])
                    elif operation == "encoding":
                        result = self._handle_encoding(current_df, operations[operation])
                    elif operation == "typo_fix":
                        result = self._handle_typo_fix(current_df, operations[operation])
                    elif operation == "normalization":
                        result = self._handle_normalization(current_df, operations[operation])
                    else:
                        continue
                    
                    # Sanitize result for JSON compliance
                    sanitized_result = self._sanitize_operation_result(result)
                    pipeline_results["operations"].append({operation: sanitized_result})
                    
                    if result["status"] == "success":
                        current_df = result["data"]
                        current_df = clean_dataframe_for_processing(current_df)
                        logging.info(f"{operation} operation completed successfully")
                    else:
                        error_msg = f"{operation} operation failed: {result.get('message', 'Unknown error')}"
                        logging.error(error_msg)
                        pipeline_results["errors"].append(error_msg)
                        # Continue with next operation instead of stopping
                        
                except Exception as e:
                    error_msg = f"{operation} operation failed with exception: {str(e)}"
                    logging.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
                    # Continue with next operation
            
            # Set final data shape
            pipeline_results["final_data_shape"] = [int(current_df.shape[0]), int(current_df.shape[1])]
            logging.info(f"Final data shape: {current_df.shape}")
            
            # Save final result if output path provided
            if output_path:
                try:
                    final_df = clean_dataframe_for_processing(current_df)
                    final_df.to_csv(output_path, index=False)
                    pipeline_results["output_path"] = output_path
                    pipeline_results["data_saved"] = True
                    logging.info(f"Final data saved to: {output_path}")
                except Exception as e:
                    error_msg = f"Failed to save final data: {str(e)}"
                    logging.error(error_msg)
                    pipeline_results["errors"].append(error_msg)
            
            pipeline_results["end_time"] = datetime.now().isoformat()
            
            # Calculate processing time
            try:
                start_time = datetime.fromisoformat(pipeline_results["start_time"])
                end_time = datetime.fromisoformat(pipeline_results["end_time"])
                processing_time = (end_time - start_time).total_seconds()
                pipeline_results["total_processing_time"] = processing_time
            except Exception:
                pipeline_results["total_processing_time"] = None
            
            logging.info(f"Pipeline completed. Final shape: {current_df.shape}")
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline failed with unexpected error: {str(e)}"
            logging.error(error_msg)
            pipeline_results["status"] = "error"
            pipeline_results["errors"].append(error_msg)
            pipeline_results["end_time"] = datetime.now().isoformat()
            return pipeline_results
    
    def _sanitize_operation_result(self, result):
        """Sanitize operation result by removing DataFrame and cleaning data."""
        sanitized = result.copy()
        
        # Remove DataFrame from result
        if 'data' in sanitized:
            del sanitized['data']
        
        # Sanitize all other values
        return sanitize_for_json(sanitized)
    
    def _handle_missing_values(self, df: pd.DataFrame, config: dict):
        """Handle missing values operation."""
        try:
            strategy = config.get("strategy", "drop_rows")
            threshold = config.get("threshold", 0.5)
            
            return self.missing_values_handler.fix_missing_values(
                df=df,
                strategy=strategy,
                threshold=threshold
            )
        except Exception as e:
            logging.error(f"Error in _handle_missing_values: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_duplicates(self, df: pd.DataFrame, config: dict):
        """Handle duplicate removal operation."""
        try:
            return self.duplicate_handler.fix_duplicate(df=df)
        except Exception as e:
            logging.error(f"Error in _handle_duplicates: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_outliers(self, df: pd.DataFrame, config: dict):
        """Handle outlier detection and treatment operation."""
        try:
            method = config.get("method", "iqr")
            columns = config.get("columns", None)
            action = config.get("action", "remove")
            threshold = config.get("threshold", 3)
            
            result = handle_outliers(
                df=df,
                columns=columns,
                method=method,
                action=action,
                threshold=threshold
            )
            return result
        except Exception as e:
            logging.error(f"Error in _handle_outliers: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_data_type_conversion(self, df: pd.DataFrame, config: dict):
        """Handle data type conversion operation."""
        try:
            type_mapping = config.get("type_mapping", {})
            auto_detect = config.get("auto_detect", True)
            errors = config.get("errors", "coerce")
            
            result = convert_data_types(
                df=df,
                type_mapping=type_mapping,
                auto_detect=auto_detect,
                errors=errors
            )
            return result
        except Exception as e:
            logging.error(f"Error in _handle_data_type_conversion: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_text_cleaning(self, df: pd.DataFrame, config: dict):
        """Handle text cleaning operation."""
        try:
            columns = config.get("columns", None)
            operations = config.get("operations", ["lowercase", "remove_whitespace"])
            custom_patterns = config.get("custom_patterns", None)
            
            result = clean_text_columns(
                df=df,
                columns=columns,
                operations=operations,
                custom_patterns=custom_patterns
            )
            return result
        except Exception as e:
            logging.error(f"Error in _handle_text_cleaning: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_datetime_parsing(self, df: pd.DataFrame, config: dict):
        """Handle datetime parsing operation."""
        try:
            columns = config.get("columns", None)
            date_format = config.get("date_format", None)
            auto_detect = config.get("auto_detect", True)
            extract_features = config.get("extract_features", False)
            errors = config.get("errors", "coerce")
            
            result = parse_datetime_columns(
                df=df,
                columns=columns,
                date_format=date_format,
                auto_detect=auto_detect,
                extract_features=extract_features,
                errors=errors
            )
            return result
        except Exception as e:
            logging.error(f"Error in _handle_datetime_parsing: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_encoding(self, df: pd.DataFrame, config: dict):
        """Handle encoding operation."""
        try:
            method = config.get("method", "label")
            columns = config.get("columns", None)
            drop_first = config.get("drop_first", False)
            
            return self.encoding_handler.encode_categorical_data(
                df=df,
                columns=columns,
                method=method,
                drop_first=drop_first
            )
        except Exception as e:
            logging.error(f"Error in _handle_encoding: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_typo_fix(self, df: pd.DataFrame, config: dict):
        """Handle typo correction operation."""
        try:
            method = config.get("method", "common_typos")
            columns = config.get("columns", None)
            similarity_threshold = config.get("similarity_threshold", 80)
            custom_dict = config.get("custom_dict", None)
            
            return self.typo_handler.fix_typos(
                df=df,
                columns=columns,
                method=method,
                similarity_threshold=similarity_threshold,
                custom_dict=custom_dict
            )
        except Exception as e:
            logging.error(f"Error in _handle_typo_fix: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def _handle_normalization(self, df: pd.DataFrame, config: dict):
        """Handle normalization operation."""
        try:
            method = config.get("method", "standard")
            columns = config.get("columns", None)
            feature_range = config.get("feature_range", (0, 1))
            with_mean = config.get("with_mean", True)
            with_std = config.get("with_std", True)
            
            return self.normalization_handler.normalize_data(
                df=df,
                columns=columns,
                method=method,
                feature_range=feature_range,
                with_mean=with_mean,
                with_std=with_std
            )
        except Exception as e:
            logging.error(f"Error in _handle_normalization: {e}")
            return {"status": "error", "message": str(e), "updates": []}
    
    def get_dataset_info(self, file_path: str):
        """Get basic information about the dataset with improved missing value detection."""
        try:
            df = pd.read_csv(file_path)
            
            # Basic info
            shape = df.shape
            columns = df.columns.tolist()
            
            # Data types - convert to JSON serializable format
            dtypes_dict = {}
            for col, dtype in df.dtypes.items():
                dtypes_dict[str(col)] = str(dtype)
            
            # Missing values - use multiple methods to ensure detection
            missing_values_dict = {}
            total_missing = 0
            
            for col in df.columns:
                # Count different types of missing values
                null_count = df[col].isnull().sum()  # Standard NaN detection
                na_count = df[col].isna().sum()      # Alternative NaN detection
                empty_strings = (df[col].astype(str) == '').sum()  # Empty strings
                whitespace_only = df[col].astype(str).str.strip().eq('').sum()  # Whitespace only
                
                # For object columns, also check for common missing value representations
                if df[col].dtype == 'object':
                    common_missing = df[col].astype(str).str.lower().isin([
                        'nan', 'null', 'none', 'na', 'n/a', 'missing', 'unknown', 
                        '', ' ', 'nil', 'undefined'
                    ]).sum()
                    
                    # Take the maximum of all missing value counts
                    total_missing_col = max(null_count, na_count, empty_strings, 
                                          whitespace_only, common_missing)
                else:
                    total_missing_col = max(null_count, na_count)
                
                missing_values_dict[str(col)] = int(total_missing_col)
                total_missing += total_missing_col
            
            # Duplicates
            duplicate_count = df.duplicated().sum()
            
            # Memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            
            # Sample data - first 5 rows with proper cleaning
            sample_data = []
            for _, row in df.head(5).iterrows():
                cleaned_row = {}
                for col, val in row.items():
                    # Handle different types of values for JSON serialization
                    if pd.isna(val):
                        cleaned_row[col] = None
                    elif isinstance(val, (int, np.integer)):
                        cleaned_row[col] = int(val)
                    elif isinstance(val, (float, np.floating)):
                        if math.isnan(val) or math.isinf(val):
                            cleaned_row[col] = None
                        else:
                            cleaned_row[col] = float(val)
                    else:
                        cleaned_row[col] = str(val)
                sample_data.append(cleaned_row)
            
            # Create info dictionary
            info = {
                "shape": [int(shape[0]), int(shape[1])],
                "columns": columns,
                "dtypes": dtypes_dict,
                "missing_values": missing_values_dict,
                "total_missing": int(total_missing),
                "duplicates": int(duplicate_count),
                "memory_usage": int(memory_usage),
                "sample_data": sample_data
            }
            
            logging.info(f"Dataset info generated - Shape: {shape}, Missing: {total_missing}, Duplicates: {duplicate_count}")
            
            return {"status": "success", "info": info}
            
        except Exception as e:
            error_msg = f"Error in get_dataset_info: {e}"
            logging.error(error_msg)
            return {"status": "error", "message": str(e)}
    
    def validate_operations(self, operations: dict):
        """Validate the operations configuration."""
        valid_operations = ["missing_values", "duplicates", "outliers", "data_type_conversion", 
                           "text_cleaning", "datetime_parsing", "encoding", "typo_fix", "normalization"]
        errors = []
        
        for op_name, op_config in operations.items():
            if op_name not in valid_operations:
                errors.append(f"Unknown operation: {op_name}")
                continue
            
            if not isinstance(op_config, dict):
                errors.append(f"Operation {op_name} must be a dictionary")
                continue
            
            if not op_config.get("enabled", False):
                continue
            
            # Basic validation for each operation
            if op_name == "missing_values":
                strategy = op_config.get("strategy", "drop_rows")
                valid_strategies = ["drop_rows", "drop_rows_threshold", "drop_columns", 
                                  "drop_columns_threshold", "fill_mean", "fill_median", 
                                  "fill_mode", "forward_fill", "backward_fill"]
                if strategy not in valid_strategies:
                    errors.append(f"Invalid missing values strategy: {strategy}")
            
            elif op_name == "outliers":
                method = op_config.get("method", "iqr")
                valid_methods = ["iqr", "zscore", "modified_zscore", "isolation_forest"]
                if method not in valid_methods:
                    errors.append(f"Invalid outliers method: {method}")
        
        return {"valid": len(errors) == 0, "errors": errors}