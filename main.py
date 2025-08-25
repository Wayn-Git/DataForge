from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import tempfile
import json
from typing import Optional, Dict, Any
import sys
import traceback

# Add the current directory and pipeline directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.join(current_dir, 'pipeline')
if os.path.exists(pipeline_dir):
    sys.path.append(pipeline_dir)
else:
    # Try parent directory structure
    parent_dir = os.path.dirname(current_dir)
    pipeline_dir = os.path.join(parent_dir, 'pipeline')
    if os.path.exists(pipeline_dir):
        sys.path.append(pipeline_dir)

from pipeline import DataCleaningPipeline

app = FastAPI(
    title="Data Cleaning Pipeline API",
    description="API for cleaning data using the DataCleaningPipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = DataCleaningPipeline()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def clean_dataset_info_for_json(dataset_info):
    """Clean dataset info to ensure JSON serialization compatibility."""
    import math
    
    cleaned = {}
    for key, value in dataset_info.items():
        if key == "sample_data":
            # Clean sample data rows
            cleaned_rows = []
            for row in value:
                cleaned_row = {}
                for col, val in row.items():
                    if isinstance(val, (int, float)):
                        if math.isnan(val) or math.isinf(val):
                            cleaned_row[col] = None
                        else:
                            cleaned_row[col] = val
                    else:
                        cleaned_row[col] = val
                cleaned_rows.append(cleaned_row)
            cleaned[key] = cleaned_rows
        elif isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                cleaned[key] = 0
            else:
                cleaned[key] = value
        else:
            cleaned[key] = value
    
    return cleaned

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Data Cleaning Pipeline API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "pipeline": "ready"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file for processing."""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save file to uploads directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get dataset info
        info = pipeline.get_dataset_info(file_path)
        
        # Clean the dataset info to ensure JSON serialization
        if info["status"] == "success" and info.get("info"):
            # Ensure all numeric values are finite
            cleaned_info = clean_dataset_info_for_json(info["info"])
        else:
            cleaned_info = None
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": file_path,
            "dataset_info": cleaned_info,
            "error": info.get("message") if info["status"] == "error" else None
        }
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/clean-data")
async def clean_data(
    file_path: str = Form(...),
    operations: str = Form(...)  # JSON string of operations
):
    """Run the data cleaning pipeline."""
    try:
        print(f"Received clean-data request for file: {file_path}")
        print(f"Operations string: {operations}")
        
        # Parse operations from JSON string
        operations_dict = json.loads(operations)
        print(f"Parsed operations dict: {operations_dict}")
        
        # Validate operations
        validation = pipeline.validate_operations(operations_dict)
        if not validation["valid"]:
            error_msg = f"Invalid operations: {validation['errors']}"
            print(f"Validation error: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(UPLOAD_DIR, f"{base_name}_cleaned.csv")
        
        # Run pipeline
        print(f"Starting pipeline with operations: {operations_dict}")
        result = pipeline.run_pipeline(
            file_path=file_path,
            operations=operations_dict,
            output_path=output_path
        )
        
        print(f"Pipeline result status: {result['status']}")
        if result.get("errors"):
            print(f"Pipeline errors: {result['errors']}")
        
        if result["status"] == "success":
            return {
                "status": "success",
                "message": "Data cleaning completed successfully",
                "result": result,
                "output_file": output_path,
                "download_url": f"/download/{os.path.basename(output_path)}"
            }
        else:
            error_msg = f"Pipeline failed: {result.get('errors', ['Unknown error'])}"
            print(f"Pipeline error: {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
            
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in operations parameter: {str(e)}"
        print(f"JSON decode error: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Data cleaning failed: {str(e)}"
        print(f"Unexpected error: {error_msg}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed file."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv"
    )

@app.get("/files")
async def list_files():
    """List all uploaded and processed files."""
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                files.append({
                    "filename": filename,
                    "size_bytes": file_size,
                    "size_mb": round(file_size / (1024 * 1024), 2),
                    "is_cleaned": "_cleaned" in filename
                })
        
        return {"status": "success", "files": files}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a file."""
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        os.remove(file_path)
        return {"status": "success", "message": f"File {filename} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.get("/pipeline-info")
async def get_pipeline_info():
    """Get information about available pipeline operations."""
    return {
        "status": "success",
        "operations": {
            "missing_values": {
                "description": "Handle missing values in the dataset",
                "strategies": [
                    "drop_rows", "drop_rows_threshold", "drop_columns", 
                    "drop_columns_threshold", "fill_mean", "fill_median", 
                    "fill_mode", "forward_fill", "backward_fill"
                ],
                "parameters": {
                    "strategy": "string (required)",
                    "threshold": "float (0.0-1.0, optional)"
                }
            },
            "duplicates": {
                "description": "Remove duplicate rows from the dataset",
                "parameters": {}
            },
            "outliers": {
                "description": "Handle outliers in numeric columns",
                "methods": ["iqr", "zscore", "modified_zscore", "isolation_forest"],
                "actions": ["remove", "cap", "transform"],
                "parameters": {
                    "method": "string (required)",
                    "action": "string (required)",
                    "threshold": "float (optional)",
                    "columns": "list (optional)"
                }
            },
            "data_type_conversion": {
                "description": "Convert data types automatically or with custom mapping",
                "parameters": {
                    "auto_detect": "boolean (optional)",
                    "type_mapping": "dict (optional)",
                    "errors": "string (optional)"
                }
            },
            "text_cleaning": {
                "description": "Clean text columns with various operations",
                "operations": ["lowercase", "uppercase", "remove_whitespace", "remove_punctuation", "remove_numbers", "remove_special_chars"],
                "parameters": {
                    "operations": "list (required)",
                    "columns": "list (optional)",
                    "custom_patterns": "dict (optional)"
                }
            },
            "datetime_parsing": {
                "description": "Parse datetime columns and extract features",
                "parameters": {
                    "columns": "list (optional)",
                    "date_format": "string (optional)",
                    "auto_detect": "boolean (optional)",
                    "extract_features": "boolean (optional)",
                    "errors": "string (optional)"
                }
            },
            "encoding": {
                "description": "Encode categorical variables",
                "methods": ["label", "onehot", "target"],
                "parameters": {
                    "method": "string (required)",
                    "columns": "list (optional)",
                    "drop_first": "boolean (optional)"
                }
            },
            "typo_fix": {
                "description": "Fix typos and spelling errors",
                "methods": ["common_typos", "fuzzy_match", "spell_check"],
                "parameters": {
                    "method": "string (required)",
                    "columns": "list (optional)",
                    "similarity_threshold": "integer (optional)",
                    "custom_dict": "dict (optional)"
                }
            },
            "normalization": {
                "description": "Normalize numerical data",
                "methods": ["standard", "minmax", "robust", "normalize"],
                "parameters": {
                    "method": "string (required)",
                    "columns": "list (optional)",
                    "feature_range": "tuple (optional)",
                    "with_mean": "boolean (optional)",
                    "with_std": "boolean (optional)"
                }
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)