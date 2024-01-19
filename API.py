import requests
import pandas as pd
from dateutil import parser

# ClinicalTrials.gov API endpoint
api_url = "https://clinicaltrials.gov/api/query/study_fields?"

# API parameters for searching pain medication trials
search_term = "PAIN AND SEACRH[RECRUITING,ACTIVE]"
fields = "NCTId,Condition,InterventionName,StartDate,CompletionDate,OverallStatus"



# Construct the API query
query_params = {
    "expr": search_term,
    "fields": fields,
    "min_rnk": 1,
    #"max_rnk": 5,  # Adjust as needed to limit the number of results
    "fmt": "json",

}

response = requests.get(api_url, params=query_params)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    clinical_trials_data = response.json()

    # Create a simplified time series-like dataset
    dataset = []
    for study in clinical_trials_data["StudyFieldsResponse"]["StudyFields"]:
        nct_id = study.get("NCTId", "")
        condition = study.get("Condition", "")
        intervention = study.get("InterventionName", "")
        start_dates = study.get("StartDate", [])
        completion_dates = study.get("CompletionDate", [])
        overall_status = study.get("OverallStatus", "")

        # Convert start dates to datetime (if it's a list)
        start_date_objects = []
        if isinstance(start_dates, list):
            for date in start_dates:
                try:
                    date_object = parser.parse(date)
                except ValueError:
                    print(f"Skipping invalid start date: {date}")
                    continue
                start_date_objects.append(date_object)
        else:
            try:
                date_object = parser.parse(start_dates)
                start_date_objects.append(date_object)
            except ValueError:
                print(f"Skipping invalid start date: {start_dates}")

        # Convert completion dates to datetime (if it's a list)
        completion_date_objects = []
        if isinstance(completion_dates, list):
            for date in completion_dates:
                try:
                    date_object = parser.parse(date)
                except ValueError:
                    print(f"Skipping invalid completion date: {date}")
                    continue
                completion_date_objects.append(date_object)
        else:
            try:
                date_object = parser.parse(completion_dates)
                completion_date_objects.append(date_object)
            except ValueError:
                print(f"Skipping invalid completion date: {completion_dates}")

        # Append data to the dataset
        dataset.append({
            "NCTId": nct_id,
            "Condition": condition,
            "Intervention": intervention,
            "StartDates": start_date_objects,
            "CompletionDates": completion_date_objects,
            "OverallStatus": overall_status,
        })

    # Create a DataFrame for further analysis
    df = pd.DataFrame(dataset)

    # Display the DataFrame
    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    print(df)

else:
    print("Error in API request:", response.status_code)