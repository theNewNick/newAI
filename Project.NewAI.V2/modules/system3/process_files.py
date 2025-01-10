# modules/system3/process_files.py

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def process_uploaded_file_system3(file_path, file_type):
    """
    A local CSV (or XLS/XLSX) reading & parsing function for System3.
    Returns a dict of the extracted fields or None on error.

    :param file_path: full path to the uploaded file on disk
    :param file_type: a string identifier like 'income_statement', 'balance_sheet', or 'cash_flow_statement'
    :return: dictionary of parsed data (key/value) or None if something fails
    """
    try:
        # Identify extension
        ext = os.path.splitext(file_path)[1].lower()

        # Read file with pandas
        if ext == ".csv":
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded CSV file from {file_path} for {file_type}")
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
            logger.debug(f"Loaded Excel file from {file_path} for {file_type}")
        else:
            logger.error(f"Unsupported file extension: {ext}")
            return None

        # Example fill-nas
        df = df.fillna(0)

        # Below is a simplistic approach: 
        #   - Column 0 = label
        #   - Column 1 = numeric value
        #   - We create a dictionary {label: value}
        # Adjust as needed based on your actual data format.

        labels = df.iloc[:, 0].astype(str).str.strip().tolist()
        # Attempt to convert second column to float. 
        # If your data has multiple columns, adapt the logic or iterate columns as needed.
        if df.shape[1] < 2:
            logger.error(f"Data for {file_type} has fewer than 2 columns, cannot parse.")
            return None

        values = df.iloc[:, 1].apply(_safe_float_convert).tolist()
        
        # Build dictionary from label -> value
        parsed_data = {}
        for label, val in zip(labels, values):
            parsed_data[label] = val

        logger.debug(f"Parsed data from {file_type}: {parsed_data}")

        return parsed_data

    except Exception as e:
        logger.exception(f"Error processing file for {file_type} at path '{file_path}'")
        return None

    finally:
        # Cleanup: remove file to avoid leftover temp files
        if os.path.exists(file_path):
            os.remove(file_path)

def _safe_float_convert(x):
    """
    Attempts to convert a value to float, stripping commas, parentheses, etc.
    Returns 0.0 if conversion fails.
    """
    try:
        s = str(x).replace(',', '').replace('(', '-').replace(')', '')
        return float(s)
    except:
        return 0.0
