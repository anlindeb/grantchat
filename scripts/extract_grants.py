import pandas as pd
import json
import os
import glob # Import glob to find files matching a pattern
import numpy as np
import re # Import regex module for more flexible searching

# Configuration
# --- Input ---
# Directory containing the CSV files
CSV_DIRECTORY = "." # Current directory. Change if files are elsewhere.
# Pattern to match the CSV files
CSV_PATTERN = os.path.join(CSV_DIRECTORY, "grants-search-*.csv")

# --- Filtering ---
# Eligibility text to search for (case-insensitive)
ELIGIBILITY_TEXT_TARGET = "independent_school_districts"
# Opportunity statuses to include (case-insensitive comparison used below)
OPPORTUNITY_STATUS_TARGETS = ["posted", "forecasted"] # Include both statuses

# --- Output ---
OUTPUT_DIR = "grant_data"
# Updated output filename to reflect the source pattern
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "independent_school_district_grants_search_combined.json")

# --- CSV Column Mapping (Updated based on user-provided headers) ---
# Maps the actual CSV header names to the desired JSON keys.
COLUMN_MAPPING = {
    # Actual CSV Header Name : Target JSON Key
    "opportunity_id": "opportunityID",
    "opportunity_title": "opportunityTitle",
    "opportunity_number": "opportunityNumber",
    "agency_name": "agencyName",
    "summary_description": "description", # Using summary_description for description
    "post_date": "postDate",
    "close_date": "closeDate",
    "close_date_description": "closeDateExplanation",
    "estimated_total_program_funding": "estimatedFunding",
    "award_ceiling": "awardCeiling",
    "award_floor": "awardFloor",
    "applicant_types": "eligibilityCodes", # This column contains text like 'independent_school_districts'
    "applicant_eligibility_description": "eligibilityDesc",
    "opportunity_assistance_listings": "cfdaNumbers", # Assuming this contains CFDA numbers
    "category": "opportunityCategory",
    "funding_instruments": "fundingInstrumentType",
    "opportunity_status": "opportunityStatus", # Adding status to mapping if needed later
    # Add other relevant fields from the CSV if desired for the LLM
    "additional_info_url": "additionalInfoUrl",
    "expected_number_of_awards": "expectedAwards",
}

# --- Key Column Names (Updated based on user-provided headers) ---
# These MUST match the actual headers in your CSV file.
ELIGIBILITY_COLUMN_NAME = "applicant_types" # Column containing eligibility text
STATUS_COLUMN_NAME = "opportunity_status"
OPPORTUNITY_ID_COLUMN = "opportunity_id"


def process_grants_search_csvs(csv_pattern):
    """
    Finds CSV files matching a pattern, reads grant data, filters, combines,
    deduplicates, formats, and returns a list of relevant grants.
    """
    csv_files = glob.glob(csv_pattern) # Find files matching the pattern
    if not csv_files:
        print(f"No CSV files found matching pattern: {csv_pattern}")
        return []

    print(f"Found {len(csv_files)} CSV files to process:")
    for f in csv_files:
        print(f"- {f}")

    all_relevant_grants = []
    processed_opportunity_ids = set() # Keep track of processed IDs for deduplication

    for csv_path in csv_files:
        print(f"\nProcessing file: {csv_path}")
        try:
            # Read the CSV file, explicitly setting dtype to str for all columns initially
            df = pd.read_csv(csv_path, dtype=str, on_bad_lines='skip')
            print(f"Successfully read {len(df)} rows from {csv_path}.")
            df_columns = df.columns

            # --- Verify Essential Columns Exist ---
            if STATUS_COLUMN_NAME not in df_columns:
                 print(f"Warning: Status column '{STATUS_COLUMN_NAME}' not found in {csv_path}. Skipping status filter for this file.")
                 df_filtered = df.copy()
            else:
                 # Filter by Opportunity Status (case-insensitive, check if status is in the target list)
                 # Fill NaN/None with empty string before filtering
                 # Convert status column to lowercase and check if it's in the list of target statuses
                 status_filter_mask = df[STATUS_COLUMN_NAME].fillna('').str.lower().isin(OPPORTUNITY_STATUS_TARGETS)
                 df_filtered = df[status_filter_mask].copy()
                 # Update log message to reflect multiple statuses
                 print(f"Filtered to {len(df_filtered)} grants with status in {OPPORTUNITY_STATUS_TARGETS}.")

            if ELIGIBILITY_COLUMN_NAME not in df_columns:
                 print(f"Warning: Eligibility column '{ELIGIBILITY_COLUMN_NAME}' not found in {csv_path}. Skipping eligibility filter for this file.")
            else:
                 # Filter by Eligibility Text (case-insensitive substring search)
                 target_text = str(ELIGIBILITY_TEXT_TARGET)
                 # Fill NaN/None with empty string before applying string operations
                 eligibility_mask = df_filtered[ELIGIBILITY_COLUMN_NAME].fillna('').astype(str).str.contains(target_text, case=False, regex=False, na=False)
                 df_filtered = df_filtered[eligibility_mask].copy()
                 print(f"Filtered to {len(df_filtered)} grants potentially matching eligibility text '{ELIGIBILITY_TEXT_TARGET}' in '{ELIGIBILITY_COLUMN_NAME}'.")

            if OPPORTUNITY_ID_COLUMN not in df_columns:
                print(f"Warning: Opportunity ID column '{OPPORTUNITY_ID_COLUMN}' not found in {csv_path}. Deduplication might not work correctly for this file.")

            # --- Data Transformation ---
            print(f"Processing {len(df_filtered)} filtered grants from {csv_path}...")
            for index, row in df_filtered.iterrows():
                grant_details = {}
                opportunity_id = None

                # Map columns based on the updated COLUMN_MAPPING
                for csv_col, json_key in COLUMN_MAPPING.items():
                    if csv_col in df_columns: # Check if the source column exists
                        value = row[csv_col]
                        # Convert potential pandas/numpy NaN/NaT to None for JSON
                        if pd.isna(value):
                             grant_details[json_key] = None
                        else:
                             grant_details[json_key] = str(value).strip() # Ensure string and strip whitespace

                        # Store opportunity ID for deduplication check
                        if csv_col == OPPORTUNITY_ID_COLUMN:
                            opportunity_id = grant_details[json_key]
                    else:
                        grant_details[json_key] = None # Assign None if source column doesn't exist

                # Skip if OpportunityID is missing or empty after stripping
                if not opportunity_id:
                     opp_id_value_raw = row.get(OPPORTUNITY_ID_COLUMN) # Get raw value
                     if pd.isna(opp_id_value_raw) or not str(opp_id_value_raw).strip():
                          print(f"Warning: Skipping row {index+2} in {csv_path} due to missing or empty '{OPPORTUNITY_ID_COLUMN}'.")
                          continue
                     else: # Attempt to recover if mapping failed but column has value
                          opportunity_id = str(opp_id_value_raw).strip()
                          if "opportunityID" not in grant_details or not grant_details["opportunityID"]:
                              grant_details["opportunityID"] = opportunity_id

                # Final check on opportunity_id before deduplication check
                if not opportunity_id:
                     print(f"Warning: Still skipping row {index+2} in {csv_path} after attempting to recover OpportunityID.")
                     continue

                # Deduplication check
                if opportunity_id in processed_opportunity_ids:
                    continue # Skip duplicate

                # Special handling for eligibility codes/text: Split the text field
                eligibility_value = grant_details.get("eligibilityCodes") # This now holds the text string
                if eligibility_value and isinstance(eligibility_value, str):
                     # Using semicolon based on user example
                     codes = [code.strip() for code in pd.Series(eligibility_value).str.split(';').explode().tolist() if code.strip()]
                     grant_details["eligibilityCodes"] = codes if codes else []
                elif "eligibilityCodes" not in grant_details: # Ensure key exists
                     grant_details["eligibilityCodes"] = []


                # Generate link using the opportunity ID
                grant_details["link"] = f"https://simpler.grants.gov/opportunity//{opportunity_id}" if opportunity_id else None

                # Ensure all target JSON keys exist, even if the source column was missing
                for mapped_csv_col, json_key in COLUMN_MAPPING.items():
                    if json_key not in grant_details:
                        grant_details[json_key] = None

                all_relevant_grants.append(grant_details)
                processed_opportunity_ids.add(opportunity_id) # Add ID to set for deduplication

        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            continue
        except pd.errors.EmptyDataError:
            print(f"Error: CSV file is empty: {csv_path}")
            continue
        except Exception as e:
            print(f"An unexpected error occurred during processing {csv_path}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            continue

    print(f"\nProcessing complete. Total unique relevant grants found across all files: {len(all_relevant_grants)}")
    return all_relevant_grants

def save_data_to_json(data, filename):
    """Saves the provided data list to a JSON file."""
    if not data:
        print("No data to save.")
        return
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved data to {filename}")
    except IOError as e:
        print(f"Error saving data to file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Pass the pattern to the processing function
    extracted_grants = process_grants_search_csvs(CSV_PATTERN)
    if extracted_grants:
        save_data_to_json(extracted_grants, OUTPUT_FILE)
    else:
        print("No grant data was processed or saved.")
