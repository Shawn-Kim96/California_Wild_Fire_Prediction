import pandas as pd


def add_mean_value(data: pd.DataFrame) -> pd.DataFrame:
    data['ratio_of_new'] = data['DayAirTmpAvg01']/data['DayPrecip14']
    return data
