'''
PART 2: Pre-processing
- Take the time to understand the data before proceeding
- Load `pred_universe_raw.csv` into a dataframe and `arrest_events_raw.csv` into a dataframe
- Perform a full outer join/merge on 'person_id' into a new dataframe called `df_arrests`
- Create a column in `df_arrests` called `y` which equals 1 if the person was arrested for a felony crime in the 365 days after their arrest date in `df_arrests`. 
- - So if a person was arrested on 2016-09-11, you would check to see if there was a felony arrest for that person between 2016-09-12 and 2017-09-11.
- - Use a print statment to print this question and its answer: What share of arrestees in the `df_arrests` table were rearrested for a felony crime in the next year?
- Create a predictive feature for `df_arrests` that is called `current_charge_felony` which will equal one if the current arrest was for a felony charge, and 0 otherwise. 
- - Use a print statment to print this question and its answer: What share of current charges are felonies?
- Create a predictive feature for `df_arrests` that is called `num_fel_arrests_last_year` which is the total number arrests in the one year prior to the current charge. 
- - So if someone was arrested on 2016-09-11, then you would check to see if there was a felony arrest for that person between 2015-09-11 and 2016-09-10.
- - Use a print statment to print this question and its answer: What is the average number of felony arrests in the last year?
- Print the mean of 'num_fel_arrests_last_year' -> pred_universe['num_fel_arrests_last_year'].mean()
- Print pred_universe.head()
- Return `df_arrests` for use in main.py for PART 3; if you can't figure this out, save as a .csv in `data/` and read into PART 3 in main.py
'''

# import the necessary packages

import pandas as pd
import os
from pathlib import Path



# -----------------------------
# Merge
# -----------------------------


def df_arrests_merge() -> pd.DataFrame:
    """
    Loads and merges the two CSV files.

    Returns:
        pd.DataFrame: Merged dataframe with datetime columns.
    """


    felony_df =  pd.read_csv('../data/pred_universe_raw.csv')
    events_df = pd.read_csv('../data/arrest_events_raw.csv')
    df_arrests = pd.merge(felony_df, events_df, on='person_id', how='outer')
    df_arrests['arrest_date_event'] = pd.to_datetime(df_arrests['arrest_date_event'])
    df_arrests['arrest_date_univ'] = pd.to_datetime(df_arrests['arrest_date_univ'])
    return df_arrests
    
# -----------------------------
# Rearrests
# -----------------------------

def df_arrests_rearrest(df_arrests) -> pd.DataFrame:

    """
    Adds 'y' column flagging rearrest within 365 days.

    Args:
        df_arrests: Merged arrests dataframe.

    Returns:
        pd.DataFrame: Dataframe with 'y' column added.
    """

    df_arrests['diff'] = (df_arrests['arrest_date_event'] - df_arrests['arrest_date_univ']).dt.days
    df_arrests['y'] = ((df_arrests['diff'] > 0) & (df_arrests['diff'] <= 365)).astype(int)
    print(f"Share rearrested for felony within a year: {df_arrests['y'].mean() * 100:.2f}%")
    return df_arrests

# -----------------------------
# Current charges
# -----------------------------

def df_arrests_felonies(df_arrests) -> pd.DataFrame:
    """
    Adds 'current_charge_felony' column.

    Args:
        df_arrests: Merged arrests dataframe.

    Returns:
        pd.DataFrame: Dataframe with 'current_charge_felony' column added.
    """
    df_arrests['current_charge_felony'] = (df_arrests['charge_degree'] == 'felony').astype(int)
    print(f"Share of current charges that are felonies: {df_arrests['current_charge_felony'].mean() * 100:.2f}%")
    return df_arrests

# -----------------------------
# Average arrests
# -----------------------------

def df_arrests_number (df_arrests) -> pd.DataFrame:
    """
    Adds 'num_fel_arrests_last_year' column.

    Args:
        df_arrests: Merged arrests dataframe.

    Returns:
        pd.DataFrame: Dataframe with 'num_fel_arrests_last_year' column added.
    """
    df_arrests['diff'] = (df_arrests['arrest_date_event'] - df_arrests['arrest_date_univ']).dt.days
    df_arrests['prior_felony'] = ((df_arrests['diff'] >= -365) & (df_arrests['diff'] <= -1) & (df_arrests['charge_degree'] == 'felony')).astype(int)
    df_arrests['num_fel_arrests_last_year'] = df_arrests.groupby('person_id')['prior_felony'].transform('sum')
    print(f"Average number of felony arrests in the last year: {df_arrests['num_fel_arrests_last_year'].mean():.2f}")
    return df_arrests

# -----------------------------
# Wrapper to export dataframe
# -----------------------------

def wrapper():
    """
    Runs the full preprocessing pipeline.

    Returns:
        pd.DataFrame: Fully processed arrests dataframe.
    """
    df_arrests = df_arrests_merge()
    df_arrests = df_arrests_rearrest(df_arrests)
    df_arrests = df_arrests_felonies(df_arrests)
    df_arrests = df_arrests_number(df_arrests)
    print(df_arrests.head())
    return df_arrests


if __name__ == '__main__':
    df = wrapper()