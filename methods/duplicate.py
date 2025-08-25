import pandas as pd
import logging
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "duplicate_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

class Duplicate:
    def fix_duplicate(self, file_path: str = None, df: pd.DataFrame = None):
        """Cleans duplicates and returns result + status updates."""
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
                
            status_updates.append(f"Dataset loaded with {len(data)} rows")
            logging.info(f"Before cleaning, dataset has {len(data)} rows")

            # Count duplicates before removal
            duplicate_mask = data.duplicated()
            duplicate_count = duplicate_mask.sum()
            status_updates.append(f"Found {duplicate_count} duplicate rows")
            logging.info(f"Found {duplicate_count} duplicate rows")

            # Remove duplicates - create a clean copy
            cleaned_df = data.drop_duplicates().copy()
            
            # Reset index to ensure clean DataFrame
            cleaned_df = cleaned_df.reset_index(drop=True)
            
            status_updates.append(f"After cleaning, dataset has {len(cleaned_df)} rows")
            logging.info(f"After cleaning, dataset has {len(cleaned_df)} rows")

            return {
                "status": "success",
                "duplicate_count": int(duplicate_count),
                "rows_before": int(len(data)),
                "rows_after": int(len(cleaned_df)),
                "columns_before": int(len(data.columns)),
                "columns_after": int(len(cleaned_df.columns)),
                "updates": status_updates,
                "data": cleaned_df
            }

        except Exception as e:
            error_msg = f"Error in fix_duplicate: {e}"
            logging.error(error_msg)
            status_updates.append(f"Error: {error_msg}")
            return {
                "status": "error",
                "message": str(e),
                "updates": status_updates
            }