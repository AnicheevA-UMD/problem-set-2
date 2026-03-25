'''
PART 1: ETL the two datasets and save each in a folder called `data/` as .csv's
'''

import pandas as pd
import os
from pathlib import Path


# -----------------------------
# First CSV Function
# -----------------------------
def felony_data() -> Path:
     """
    Downloads the first csv file, converts to a dataframe, dropscsv in /data folder

    Returns:
        Path to the written CSV file.
    """
     #Make Folder
     os.makedirs('../data', exist_ok=True)

     #Extract
     pred_universe_raw = pd.read_csv('https://www.dropbox.com/scl/fi/69syqjo6pfrt9123rubio/universe_lab6.feather?rlkey=h2gt4o6z9r5649wo6h6ud6dce&dl=1')

     #Transform
     pred_universe_raw['arrest_date_univ'] = pd.to_datetime(pred_universe_raw.filing_date)
     pred_universe_raw.drop(columns=['filing_date'], inplace=True)

     #Load
     felony_path = '../data/pred_universe_raw.csv' 
     pred_universe_raw.to_csv(felony_path, index=False)
     return (felony_path)



# -----------------------------
# Second CSV Function
# -----------------------------

def arrest_data() -> Path:
     """
    Downloads the second csv file, converts to a dataframe, drops csv in /data folder

    Returns:
        Path to the written CSV file.
    """
     #Make Folder
     os.makedirs('../data', exist_ok=True)

     #Extract
     arrest_events_raw = pd.read_csv('https://www.dropbox.com/scl/fi/wv9kthwbj4ahzli3edrd7/arrest_events_lab6.feather?rlkey=mhxozpazqjgmo6qqahc2vd0xp&dl=1')

     #Transform
     arrest_events_raw['arrest_date_event'] = pd.to_datetime(arrest_events_raw.filing_date)
     arrest_events_raw.drop(columns=['filing_date'], inplace=True)
     
     #Load
     arrest_path = '../data/arrest_events_raw.csv'
     arrest_events_raw.to_csv(arrest_path, index=False)
     return (arrest_path)

if __name__ == "__main__":
    arrest_path = arrest_data()
    felony_path = felony_data()
    print(f"Arrest data saved to: {felony_path}")
    print(f"Arrest data saved to: {arrest_path}")
