import pandas as pd

sheet_name = "Zeitreihen0h15"

frames = []

for year in range(2009, 2024):

    path = r"swissgrid/EnergieUebersichtCH-" + str(year) + ".xls"

    # read datasets
    df = pd.read_excel(path, sheet_name=sheet_name)

    # select timestamp and energy consumption columns, discard first row with column titles
    df = df.iloc[1:, :2]

    # rename columns as "datetime" and "consumption"
    df = df.rename(columns={df.columns[0]: "datetime", df.columns[1]: "consumption"})

    # convert datetime strings to datetime datatype
    df["datetime"] = pd.to_datetime(df["datetime"], dayfirst=(year > 2020))

    # convert consumption value strings to float
    df["consumption"] = pd.to_numeric(df["consumption"])

    # set datetime as index
    df = df.set_index("datetime")

    # aggregate hourly
    df = df.resample('h', label='right', closed='right').sum()

    # convert kWh to MWh
    df["consumption"] = df["consumption"] / 1000.0

    frames.append(df)

final = pd.concat(frames)

final.to_csv(r"swissgrid/hourly_consumption_2009-2023_MWh.csv")