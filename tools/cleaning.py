import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def fill_missing_values(df:pd.DataFrame)->pd.DataFrame:
    """
    This method fills the missing days and etc missing vlaues with the average of the
    day before and the day after.

    ex: if DayRelHumAvg02 is NaN, then we replace it with (DayRelHumAvg01+DayRelHumAvg03)/2

    input:
          our datafram

    output:
          updated dataframe

    """
    # Extract unique feature prefixes (e.g., "DayAirTmpAvg", "DayRelHumAvg")
    feature_prefixes = set(col[:-2] for col in df.columns if col !='incident_acres_burned')  # Remove last 2 digits to get feature name

    for prefix in feature_prefixes:
        # Get all columns that match this prefix
        matching_cols = sorted([col for col in df.columns if col.startswith(prefix)], key=lambda x: int(x[-2:]))
        for col in matching_cols:
            day_num = int(col[-2:])  # Extracting numeric day


            # Define column names for (X+1) and (X-2)
            next_day_col = f"{prefix}{str(day_num+1).zfill(2)}" if f"{prefix}{str(day_num+1).zfill(2)}" in df.columns else None
            prev_2_day_col = f"{prefix}{str(day_num-2).zfill(2)}" if f"{prefix}{str(day_num-2).zfill(2)}" in df.columns else None

            # Fill NaN using the formula: (X+1 + X-2) / 2 and other set rules

            #TODO ADD RULES FOR WHEN A CONTINOUS RANGE OF DAYS VALUES ARE MISSING
            #TODO ADD A MACHIN LEARNING MODULE FOR POPULATING DATA


            if next_day_col and prev_2_day_col:
                print('method invoked')
                df[col] = df[col].fillna((df[next_day_col] + df[prev_2_day_col]) / 2)
            elif next_day_col:  # If only X+1 exists
                print('method 2 invoked')
                df[col] = df[col].fillna(df[next_day_col])
            elif prev_2_day_col:  # If only X-2 exists
                print('method 3 invoked')
                df[col] = df[col].fillna(df[prev_2_day_col])


    return df

def hot_encode_time(data_main:pd.DataFrame)->pd.DataFrame:
    """
    This method takes in a dataframe, extracts date and time and then hot encodes the day, hour, and month.
     ex: 	day	 hour	month ->	day_2	day_3	day_4	day_5	day_6	day_7	day_8	...	month_3	month_4	...	month_10	month_11	month_12
            31	  11	  10  -> 	0.0	     0.0	0.0	     0.0	0.0	 0.0     0.0    ...	 0.0	0.0         	1.0	       0.0	      0.0
    input:
          our datafram with keys 'day', 'hour', and 'month'

    output:
          updated dataframe with hot encoded dates and time 

    """
    data_main['day'] =  pd.to_datetime(data_main['incident_date_created']).dt.day
    data_main['hour'] =  pd.to_datetime(data_main['incident_date_created']).dt.hour
    data_main['month'] =  pd.to_datetime(data_main['incident_date_created']).dt.month
    # Select columns to encode
    features_to_encode = ['day', 'hour', 'month']


    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' to avoid dummy variable trap, sparse=False

    encoded_features = encoder.fit_transform(data_main[features_to_encode])
    # Convert encoded features into a DataFrame
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(features_to_encode))

    # Concatenate the one-hot encoded features back to the original DataFrame
    temp_frame = pd.concat([temp_frame, encoded_df], axis=1)
    return temp_frame