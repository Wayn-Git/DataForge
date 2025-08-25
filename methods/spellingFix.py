import pandas as pd
import logging
import re
from difflib import SequenceMatcher
from collections import Counter
import os

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, "typo_fix_log.txt")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='a'
)

class TypoFix:
    def __init__(self):
        # Common typos dictionary - you can expand this
        self.common_typos = {
            'teh': 'the',
            'adn': 'and',
            'thier': 'their',
            'recieve': 'receive',
            'seperate': 'separate',
            'definately': 'definitely',
            'occured': 'occurred',
            'begining': 'beginning',
            'untill': 'until',
            'mispelled': 'misspelled',
            'accomodate': 'accommodate',
            'embarass': 'embarrass',
            'goverment': 'government',
            'liesure': 'leisure',
            'maintainance': 'maintenance',
            'necesary': 'necessary',
            'occassion': 'occasion',
            'posession': 'possession',
            'priviledge': 'privilege',
            'recomend': 'recommend',
            # Add some common data entry typos
            'unitd': 'united',
            'managment': 'management',
            'deparment': 'department',
            'devlopment': 'development',
            'busness': 'business',
            'finace': 'finance'
        }
    
    def fix_typos(self, file_path: str = None, df: pd.DataFrame = None, columns: list = None, method: str = "common_typos", 
                  similarity_threshold: int = 80, custom_dict: dict = None):
        """
        Fix typos in text columns using various methods.
        
        Args:
            file_path (str): Path to the CSV file (optional if df provided)
            df (pd.DataFrame): DataFrame to process (optional if file_path provided)
            columns (list): List of columns to check for typos (None for all text columns)
            method (str): Method to use - "common_typos", "fuzzy_match", "spell_check"
            similarity_threshold (int): Threshold for fuzzy matching (0-100)
            custom_dict (dict): Custom typo dictionary for replacements
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
            logging.info(f"Before typo fix, dataset has {len(data)} rows and {len(data.columns)} columns")

            # Identify text columns
            if columns is None:
                text_cols = data.select_dtypes(include=['object']).columns.tolist()
            else:
                text_cols = [col for col in columns if col in data.columns and data[col].dtype == 'object']

            if not text_cols:
                status_updates.append("No text columns found for typo correction")
                return {
                    "status": "success",
                    "message": "No text columns to process for typos",
                    "updates": status_updates,
                    "method_used": method,
                    "columns_processed": [],
                    "total_corrections": 0,
                    "rows_before": int(len(data)),
                    "rows_after": int(len(data)),
                    "columns_before": int(len(data.columns)),
                    "columns_after": int(len(data.columns)),
                    "data": data
                }

            status_updates.append(f"Found {len(text_cols)} text columns: {', '.join(text_cols)}")
            logging.info(f"Processing {len(text_cols)} text columns for typos")

            cleaned_df = data.copy()
            total_corrections = 0

            if method == "common_typos":
                status_updates.append("Applying common typo corrections...")
                typo_dict = self.common_typos.copy()
                
                if custom_dict:
                    typo_dict.update(custom_dict)
                    status_updates.append(f"Added {len(custom_dict)} custom typo corrections")

                for col in text_cols:
                    col_corrections = 0
                    
                    # Process each cell individually to avoid SettingWithCopyWarning
                    for idx in cleaned_df.index:
                        cell_value = cleaned_df.at[idx, col]
                        
                        if pd.isna(cell_value):
                            continue
                        
                        original_value = str(cell_value)
                        corrected_value = original_value
                        
                        # Apply word-level corrections (case insensitive)
                        for typo, correction in typo_dict.items():
                            # Use word boundaries to avoid partial matches
                            pattern = r'\b' + re.escape(typo) + r'\b'
                            if re.search(pattern, corrected_value, re.IGNORECASE):
                                corrected_value = re.sub(pattern, correction, corrected_value, flags=re.IGNORECASE)
                                col_corrections += 1
                        
                        if corrected_value != original_value:
                            cleaned_df.at[idx, col] = corrected_value
                    
                    if col_corrections > 0:
                        status_updates.append(f"Column {col}: Fixed {col_corrections} typos")
                        total_corrections += col_corrections

            elif method == "fuzzy_match":
                status_updates.append("Applying fuzzy matching corrections...")
                
                for col in text_cols:
                    col_corrections = 0
                    # Get all unique non-null values in the column
                    non_null_mask = cleaned_df[col].notna()
                    if not non_null_mask.any():
                        continue
                        
                    unique_values = cleaned_df.loc[non_null_mask, col].astype(str).unique()
                    
                    if len(unique_values) < 2:
                        continue
                    
                    # Find similar strings that might be typos
                    corrections_map = {}
                    value_counts = cleaned_df[col].value_counts()
                    
                    # Compare each value with every other value
                    for i, val1 in enumerate(unique_values):
                        val1_str = str(val1).lower().strip()
                        
                        for j, val2 in enumerate(unique_values[i+1:], i+1):
                            val2_str = str(val2).lower().strip()
                            
                            # Skip if strings are too different in length
                            if abs(len(val1_str) - len(val2_str)) > max(len(val1_str), len(val2_str)) * 0.3:
                                continue
                            
                            try:
                                similarity = SequenceMatcher(None, val1_str, val2_str).ratio() * 100
                                
                                if similarity >= similarity_threshold and val1 != val2:
                                    # Choose the more frequent value as the correct one
                                    if val1 in value_counts and val2 in value_counts:
                                        if value_counts[val1] >= value_counts[val2]:
                                            corrections_map[val2] = val1
                                        else:
                                            corrections_map[val1] = val2
                            except Exception as e:
                                status_updates.append(f"Error comparing '{val1}' and '{val2}': {str(e)}")
                                continue
                    
                    # Apply corrections
                    for old_val, new_val in corrections_map.items():
                        mask = cleaned_df[col] == old_val
                        corrections_count = mask.sum()
                        if corrections_count > 0:
                            cleaned_df.loc[mask, col] = new_val
                            col_corrections += corrections_count
                    
                    if col_corrections > 0:
                        status_updates.append(f"Column {col}: Fixed {col_corrections} fuzzy matches")
                        total_corrections += col_corrections

            elif method == "spell_check":
                status_updates.append("Applying basic spell checking...")
                
                # This is a simplified spell check using frequency analysis
                for col in text_cols:
                    col_corrections = 0
                    
                    # Get all words from the column
                    all_words = []
                    non_null_values = cleaned_df[col].dropna()
                    
                    for val in non_null_values:
                        try:
                            words = re.findall(r'\b\w+\b', str(val).lower())
                            all_words.extend(words)
                        except Exception as e:
                            status_updates.append(f"Error extracting words from '{val}': {str(e)}")
                            continue
                    
                    if not all_words:
                        continue
                    
                    word_freq = Counter(all_words)
                    # Consider words that appear more than once as "correct"
                    common_words = set([word for word, freq in word_freq.items() if freq > 1])
                    
                    # If we don't have enough common words, use the most frequent ones
                    if len(common_words) < 10:
                        common_words = set([word for word, freq in word_freq.most_common(50)])
                    
                    for idx in cleaned_df.index:
                        cell_value = cleaned_df.at[idx, col]
                        
                        if pd.isna(cell_value):
                            continue
                        
                        original_value = str(cell_value)
                        corrected_value = original_value
                        
                        try:
                            words = re.findall(r'\b\w+\b', original_value.lower())
                            for word in words:
                                if len(word) > 2 and word not in common_words:
                                    # Find the most similar word from common words
                                    best_match = None
                                    best_similarity = 0
                                    
                                    for common_word in common_words:
                                        if abs(len(word) - len(common_word)) <= 2:
                                            sim = SequenceMatcher(None, word, common_word).ratio() * 100
                                            if sim > best_similarity and sim >= similarity_threshold:
                                                best_similarity = sim
                                                best_match = common_word
                                    
                                    if best_match:
                                        pattern = r'\b' + re.escape(word) + r'\b'
                                        corrected_value = re.sub(pattern, best_match, corrected_value, flags=re.IGNORECASE)
                                        col_corrections += 1
                            
                            if corrected_value != original_value:
                                cleaned_df.at[idx, col] = corrected_value
                        except Exception as e:
                            status_updates.append(f"Error spell checking cell at index {idx}: {str(e)}")
                            continue
                    
                    if col_corrections > 0:
                        status_updates.append(f"Column {col}: Fixed {col_corrections} spelling errors")
                        total_corrections += col_corrections

            else:
                raise ValueError(f"Unknown typo fix method: {method}")

            # Ensure the DataFrame is clean and doesn't have any issues
            cleaned_df = cleaned_df.copy()

            status_updates.append(f"Typo correction completed. Total corrections made: {total_corrections}")
            status_updates.append(f"Final dataset has {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")
            logging.info(f"After typo fix, dataset has {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")

            return {
                "status": "success",
                "method_used": method,
                "columns_processed": text_cols,
                "total_corrections": int(total_corrections),
                "rows_before": int(len(data)),
                "rows_after": int(len(cleaned_df)),
                "columns_before": int(len(data.columns)),
                "columns_after": int(len(cleaned_df.columns)),
                "updates": status_updates,
                "data": cleaned_df
            }

        except Exception as e:
            error_msg = f"Error in fix_typos: {e}"
            logging.error(error_msg)
            status_updates.append(f"Error: {error_msg}")
            return {
                "status": "error",
                "message": str(e),
                "updates": status_updates
            }

    def add_custom_typos(self, typo_dict: dict):
        """Add custom typos to the common typos dictionary."""
        self.common_typos.update(typo_dict)

    def get_typo_statistics(self, df: pd.DataFrame, columns: list = None):
        """Get statistics about potential typos in the dataset."""
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        stats = {}
        for col in columns:
            if col in df.columns:
                values = df[col].dropna()
                unique_count = values.nunique()
                total_count = len(values)
                
                # Calculate potential typo indicators
                avg_length = values.astype(str).str.len().mean()
                length_variance = values.astype(str).str.len().var()
                
                stats[col] = {
                    'unique_values': int(unique_count),
                    'total_values': int(total_count),
                    'uniqueness_ratio': unique_count / total_count if total_count > 0 else 0,
                    'avg_length': float(avg_length) if not pd.isna(avg_length) else 0,
                    'length_variance': float(length_variance) if not pd.isna(length_variance) else 0
                }
        
        return stats