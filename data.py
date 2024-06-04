import os
import holidays
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset
from tqdm import tqdm

from utils.scaler import MaxScaler

dataset_names = ['en', 'spain', 'swiss-consumption', 'swiss-production']


def en(filepath='./data/en/selected_data_ISONE.csv'):
    df = pd.read_csv(filepath)

    df['date'] = df['date'].apply(pd.to_datetime)

    # New column with weekend information
    df['weekend'] = df['date'].dt.dayofweek > 4
    df['weekend'] = np.where(df['weekend'], 1, 0)

    # Find Swiss Holidays
    ch_holidays = []
    for date in holidays.CH(years=list(range(2008, 2015))).items():
        ch_holidays.append(str(date[0]))

    # New column with Holiday information
    df['holiday'] = [1 if str(val).split()[0] in ch_holidays else 0 for val in df['date']]

    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'] - 1, unit='h')

    df['date'] = df['date'].astype(str)

    df = df.rename(columns={"demand": "value"})

    df = df.set_index(df['datetime'])

    return df[["date", "value", "temperature", "weekday", "weekend", "holiday"]]


def spain(filepath_energy='./data/spain/energy_dataset.csv',
          filepath_weather='./data/spain/weather_features.csv'):
    df_energy = pd.read_csv(filepath_energy)
    df_weather = pd.read_csv(filepath_weather)

    # Fill NaNs
    df_energy = df_energy.interpolate(method='linear', axis=0)

    # Rename columns with dates
    df_energy.rename(columns={'time': 'date'}, inplace=True)
    df_weather.rename(columns={'dt_iso': 'date'}, inplace=True)

    # Drop duplicates
    df_weather.drop_duplicates(subset=['date', 'city_name'], inplace=True, keep='last')

    # Set date column as index
    df_energy.set_index('date', inplace=True)
    df_weather.set_index('date', inplace=True)

    # Group weather features by city names
    def group_cities():
        gb = df_weather.groupby('city_name')
        cities = [gb.get_group(x) for x in gb.groups]
        return cities

    # One dataframe per city
    cities = group_cities()
    barcelona = cities[0]
    bilbao = cities[1]
    madrid = cities[2]
    seville = cities[3]
    valencia = cities[4]

    # Rename columns so that they include the city name
    barcelona.columns = 'barcelona_' + barcelona.columns
    bilbao.columns = 'bilbao_' + bilbao.columns
    madrid.columns = 'madrid_' + madrid.columns
    seville.columns = 'seville_' + seville.columns
    valencia.columns = 'valencia_' + valencia.columns

    # Join all the dataframes
    df = (df_energy
          .join(barcelona, how='outer')
          .join(bilbao, how='outer')
          .join(madrid, how='outer')
          .join(seville, how='outer')
          .join(valencia, how='outer'))

    # Reset the index of the dataframe
    df.reset_index(inplace=True)

    df['date'] = df['date'].apply(pd.to_datetime, utc=True)
    # Remove UTC time from dates
    df['date'] = df['date'].dt.tz_localize(None)

    df['hour'] = df['date'].dt.hour

    # New column with Day of week information
    df['weekday'] = df['date'].dt.dayofweek

    # New column with weekend information
    df['weekend'] = df['date'].dt.dayofweek > 4
    df['weekend'] = np.where(df['weekend'], 1, 0)

    # Remove time from dates
    df['date'] = df['date'].dt.date

    # Find Swiss Holidays
    ch_holidays = []
    for date in holidays.CH(list(range(2015, 2019))).items():
        ch_holidays.append(str(date[0]))

    # New column with Holiday information
    df['holiday'] = [1 if str(val).split()[0] in ch_holidays else 0 for val in df['date']]

    df['date'] = df['date'].astype(str)

    df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'] - 1, unit='h')
    df = df.set_index(df['datetime'])

    df = df.rename(columns={"madrid_temp": "temperature", "total load actual": "value"})

    return df[["date", "value", "temperature", "weekday", "weekend", "holiday"]]


def swissgrid(energy_type):
    if energy_type == 'consumption':
        filepath = 'data/swissgrid/hourly_consumption_2009-2023_MWh.csv'
    elif energy_type == 'production':
        filepath = 'data/swissgrid/hourly_production_2009-2023_MWh.csv'

    df = pd.read_csv(filepath)

    # Structure dataset time columns as in New England (some of these values might be useless)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour

    # Create new row with consumption/production for time 2009-01-01 00:00
    new_first_row = pd.DataFrame(df.loc[0]).T
    new_first_row['hour'] = 0
    new_first_row['datetime'] = pd.to_datetime("2009-01-01 00:00:00")
    new_first_row[energy_type] = df[(df['day'] == 1) & (df['month'] == 1) & (df['hour'] == 0)][energy_type].agg(
        func='average')

    # Add new row (so that 2009-01-01 starts at 00:00 like other days instead of 01:00)
    df = pd.concat([new_first_row, df])
    df.reset_index(drop=True, inplace=True)

    # New column with weekend information
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['weekend'] = df['datetime'].dt.dayofweek > 4
    df['weekend'] = np.where(df['weekend'], 1, 0)

    # Find Swiss Holidays
    ch_holidays = []
    for date in holidays.CH(years=list(range(2009, 2025))).items():
        ch_holidays.append(str(date[0]))

    # New column with Holiday information
    df['holiday'] = [1 if str(val).split()[0] in ch_holidays else 0 for val in df['date']]

    # New column with Day of week information
    df['weekday'] = df['datetime'].dt.dayofweek

    df['date'] = df['date'].astype(str)

    df = df.set_index(df['datetime'])

    df = df.rename(columns={energy_type: "value"})

    return df[["date", "value", "weekday", "weekend", "holiday"]]


def get_dates(dataset_name, dataset_type, train_offset=28):
    start = end = None
    match dataset_name:
        case 'en':
            match dataset_type:
                case 'train':
                    start = '2007-01-01'
                    end = '2012-12-31'
                case 'val':
                    start = '2013-01-01'
                    end = '2013-12-31'
                case 'test':
                    start = '2014-01-01'
                    end = '2014-12-31'
        case 'spain':
            match dataset_type:
                case 'train':
                    start = '2015-03-01'
                    end = '2017-12-31'
                case 'val':
                    start = '2017-01-01'
                    end = '2017-12-31'
                case 'test':
                    start = '2018-01-01'
                    end = '2018-12-30'
        case 'swiss-consumption' | 'swiss-production':
            match dataset_type:
                case 'train':
                    # Offset is used because there is no data before 2009-01-01 in Swiss datasets
                    # so to use previous month information we must start from a month after
                    start = pd.to_datetime('2009-01-01') + pd.to_timedelta(train_offset, unit='d')
                    start = start.strftime('%Y-%m-%d')
                    end = '2021-12-31'
                case 'val':
                    start = '2022-01-01'
                    end = '2022-12-31'
                case 'test':
                    start = '2023-01-01'
                    end = '2024-01-01'

    return start, end


# FUNCTION TO CHECK IF WEATHER DATA IS AVAILABLE BASED ON WHICH DATASET IS TAKEN INTO ACCOUNT #

def has_temperatures(dataset_name):
    return dataset_name in ['en', 'spain']


# TORCH DATASETS #


class PDDataset(Dataset):
    def __init__(self,
                 df,
                 start_date,
                 end_date,
                 weather_is_available,
                 feature_scaler=None,
                 load_scaler=None,
                 ds_type='train',
                 method='baseline',
                 anchoring='week'):

        self.ds_type = ds_type
        self.method = method
        self.anchoring = anchoring

        date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)

        days = pd.to_datetime(df['date'][date_mask].unique(), utc=False)

        prev_days = days - pd.to_timedelta(1, unit='d')
        prev_days = prev_days.strftime('%Y-%m-%d')
        prev_weeks = days - pd.to_timedelta(7, unit='d')
        prev_weeks = prev_weeks.strftime('%Y-%m-%d')
        prev_14 = days - pd.to_timedelta(14, unit='d')
        prev_14 = prev_14.strftime('%Y-%m-%d')
        prev_months = days - pd.to_timedelta(28, unit='d')
        prev_months = prev_months.strftime('%Y-%m-%d')

        prev_35 = days - pd.to_timedelta(35, unit='d')
        prev_35 = prev_35.strftime('%Y-%m-%d')
        prev_56 = days - pd.to_timedelta(56, unit='d')
        prev_56 = prev_56.strftime('%Y-%m-%d')

        days = days.strftime('%Y-%m-%d')

        self.n_samples = len(days)
        self.df = df

        self.days = days
        self.prev_days = prev_days
        self.prev_weeks = prev_weeks
        self.prev_months = prev_months
        self.prev_14 = prev_14
        self.prev_35 = prev_35
        self.prev_56 = prev_56

        self.load = np.empty((self.n_samples, 24))

        self.weather_is_available = weather_is_available

        self.feature_len = 7 * 24 + 3 if weather_is_available else 3 * 24 + 3
        self.features = np.empty((self.n_samples, self.feature_len,))

        self.preload_samples()

        if feature_scaler is not None:
            if self.ds_type == 'train':
                self.features = feature_scaler.fit_transform(self.features)
            else:
                self.features = feature_scaler.transform(self.features)
        if load_scaler is not None:
            if self.ds_type == 'train':
                self.load = load_scaler.fit_transform(self.load)
            else:
                self.load = load_scaler.transform(self.load)
        self.load_scaler = load_scaler

    def inv_gt_function(self, pct_predictions):
        if self.load_scaler is not None:
            pct_predictions = self.load_scaler.inverse_transform(pct_predictions)

        # y = (x / a) - 1 ==> x = a * (y + 1), where a is the anchor (consumption 7 days prior)
        if self.anchoring == 'week':
            anchor = self.df.sort_index().loc[self.prev_weeks[0]:self.prev_weeks[-1]]["value"].to_numpy().reshape(
                self.n_samples, 24)
        else:
            anchor = self.df.loc[self.prev_months[0]:self.prev_months[-1]]["value"].to_numpy().reshape(self.n_samples,
                                                                                                       24)

        x = anchor * (pct_predictions + 1)
        return x

    def preload_samples(self):
        for item in tqdm(range(self.n_samples), total=self.n_samples):
            day = self.days[item]

            prev_day = self.prev_days[item]
            prev_week = self.prev_weeks[item]
            prev_month = self.prev_months[item]

            prev_14 = self.prev_14[item]
            prev_35 = self.prev_35[item]
            prev_56 = self.prev_56[item]

            load = self.df.loc[self.df['date'] == day]["value"].to_numpy()

            if self.method != 'baseline':  # anchor load
                # y = (x / a) - 1 where y is the gt, x is the day's consumption
                if self.anchoring == 'week':
                    anchor = self.df.loc[self.df['date'] == prev_week]["value"].to_numpy()
                else:
                    anchor = self.df.loc[self.df['date'] == prev_month]["value"].to_numpy()
                load = load / anchor - 1

            features = np.zeros((24 * 3,))
            features[:24] = self.df.loc[self.df['date'] == prev_day]["value"].to_numpy()
            features[24:48] = self.df.loc[self.df['date'] == prev_week]["value"].to_numpy()
            features[48:72] = self.df.loc[self.df['date'] == prev_month]["value"].to_numpy()

            if self.method != 'baseline':  # anchor features
                if self.anchoring == 'week':
                    anchors = (self.df.loc[self.df['date'] == prev_week]["value"].to_numpy(),
                               self.df.loc[self.df['date'] == prev_14]["value"].to_numpy(),
                               self.df.loc[self.df['date'] == prev_week][
                                   "value"].to_numpy())  # Must try with prev_month
                else:
                    anchors = (self.df.loc[self.df['date'] == prev_month]["value"].to_numpy(),
                               self.df.loc[self.df['date'] == prev_35]["value"].to_numpy(),
                               self.df.loc[self.df['date'] == prev_56]["value"].to_numpy())

                features[:24] = features[:24] / anchors[0] - 1
                features[24:48] = features[24:48] / anchors[1] - 1
                features[48:72] = features[48:72] / anchors[2] - 1

            if self.weather_is_available:
                weather_features = np.zeros((24 * 4,))
                weather_features[:24] = self.df.loc[self.df['date'] == day]["temperature"].to_numpy()
                weather_features[24:48] = self.df.loc[self.df['date'] == prev_day]["temperature"].to_numpy()
                weather_features[48:72] = self.df.loc[self.df['date'] == prev_week]["temperature"].to_numpy()
                weather_features[72:96] = self.df.loc[self.df['date'] == prev_month]["temperature"].to_numpy()
                features = np.concatenate((weather_features, features))

            day_info_features = np.zeros((3,))
            day_info_features[0] = self.df.loc[self.df['date'] == day]['holiday'].max()
            day_info_features[1] = self.df.loc[self.df['date'] == day]['weekend'].max()
            day_info_features[2] = self.df.loc[self.df['date'] == day]['weekday'].max()
            features = np.concatenate((features, day_info_features))

            self.load[item, :] = load
            self.features[item, :] = features

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        features = self.features[item]
        load = self.load[item]
        return np.float32(features), np.float32(load)




def get_train_val_test_split(dataset_name, method, anchoring, feature_scaler, load_scaler):
    def get_scaler_from_type(scaler_type):
        match scaler_type:
            case 'max':
                return MaxScaler()
            case 'minmax':
                return MinMaxScaler()
            case 'standard':
                return StandardScaler()
            case _:
                return None

    feature_scaler = get_scaler_from_type(feature_scaler)
    load_scaler = get_scaler_from_type(load_scaler)

    weather_is_available = has_temperatures(dataset_name)

    def dataset_file(ds_type):
        start_date, end_date = get_dates(dataset_name, ds_type)
        file = (f'./data/torch/'
                f'{dataset_name}_'
                f'{"baseline" if method=="baseline" else "anio"}_'
                f'{anchoring + "_" if method != "baseline" else ""}'
                f'{ds_type}_'
                f'{start_date}-{end_date}_'
                f'weather={weather_is_available}_'
                f'feature-scaler={feature_scaler}_'
                f'load-scaler={load_scaler}'
                f'.pt')
        return file

    if not os.path.exists(dataset_file('train')):
        match dataset_name:
            case 'en':
                df = en()
            case 'spain':
                df = spain()
            case 'swiss-consumption':
                df = swissgrid('consumption')
            case 'swiss-production':
                df = swissgrid('production')
            case _:
                df = None

    def load_dataset(ds_type):
        if not os.path.exists(dataset_file(ds_type)):
            start_date, end_date = get_dates(dataset_name, ds_type)
            dataset = PDDataset(df, start_date, end_date,
                                weather_is_available=weather_is_available,
                                feature_scaler=feature_scaler,
                                load_scaler=load_scaler,
                                ds_type=ds_type,
                                method=method,
                                anchoring=anchoring)
            torch.save(dataset, dataset_file(ds_type))
        else:
            dataset = torch.load(dataset_file(ds_type))
        return dataset

    print('Loading train set...')
    train_dataset = load_dataset('train')

    print('Loading validation set...')
    val_dataset = load_dataset('val')

    print('Loading test set...')
    test_dataset = load_dataset('test')

    return train_dataset, val_dataset, test_dataset
