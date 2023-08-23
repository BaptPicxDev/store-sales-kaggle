# Imports
import os
from os.path import (
    abspath,
    join,
    splitext,
    exists
)
from datetime import (
    datetime as dt
)
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)
import pandas as pd
from dotenv import (
    load_dotenv
)
import numpy as np
import kaggle
import warnings
import matplotlib.pyplot as plt

# Environment
pd.set_option('display.max_columns', 20)
warnings.filterwarnings("ignore")
load_dotenv(dotenv_path=join(abspath("../"), ".env"))
TRAIN_SET = "data/train.csv"
TEST_SET = "data/test.csv"
SAMPLE_SET = "data/sample_submission.csv"
TRANSACTION_SET = "data/transactions.csv"
STORES_SET = "data/stores.csv"
OIL_SET = "data/oil.csv"
HOLIDAYS_SET = "data/holidays_events.csv"


# Functions
def load_train():
    return pd.read_csv(abspath(TRAIN_SET), parse_dates=["date"], infer_datetime_format=True).drop(columns=["id"])


def load_test():
    return pd.read_csv(abspath(TEST_SET), parse_dates=["date"], infer_datetime_format=True).drop(columns=["id"])


def load_sample_submission():
    return pd.read_csv(abspath(SAMPLE_SET))


def load_transaction(min_date, max_date, stores):
    df = pd.read_csv(abspath(TRANSACTION_SET), parse_dates=["date"], infer_datetime_format=True)
    new_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([pd.date_range(start=min_date, end=max_date), stores],
                                         names=["date", "store_nbr"])
    )
    new_df["transactions"] = None
    new_df = new_df.reset_index()
    new_df = (
        new_df.merge(df, left_on=["date", "store_nbr"], right_on=["date", "store_nbr"], how="left")
              .drop(columns=["transactions_x"])
              .rename(columns={"transactions_y": "transactions"})
    )
    new_df["year"] = new_df.date.dt.year
    new_df["month"] = new_df.date.dt.month
    new_df["wd"] = new_df.date.dt.dayofweek
    new_df["transactions"] = (
        new_df.groupby(["store_nbr", "month", "wd"]).transform(lambda x: x.fillna(x.mean()))
              .transactions
    )
    new_df["transactions"] = (
        new_df.groupby(["store_nbr"]).transform(lambda x: x.fillna(x.mean()))
              .transactions
    )
    return new_df.drop(columns=["year", "month", "wd"])


def load_stores():
    return pd.read_csv(abspath(STORES_SET))


def load_oil(min_date, max_date):
    df = pd.read_csv(abspath(OIL_SET), parse_dates=["date"], infer_datetime_format=True)
    new = pd.DataFrame({"date": pd.date_range(start=min_date, end=max_date),
                        "dcoilwtico": None})
    new = (
        new.merge(df, left_on=["date"], right_on=["date"], how="left")
           .drop(columns=["dcoilwtico_x"])
           .rename(columns={"dcoilwtico_y": "dcoilwtico"})
    )
    new["year"] = new.date.dt.year
    new["week"] = new.date.dt.isocalendar().week
    new["dcoilwtico"] = new.groupby(["year", "week"]).transform(lambda x: x.fillna(x.mean())).dcoilwtico
    new["dcoilwtico"] = new.dcoilwtico.fillna(new.dcoilwtico.mean())
    return new.drop(columns=["year", "week"])


def load_holidays(min_date, max_date):
    df = pd.read_csv(abspath(HOLIDAYS_SET), parse_dates=["date"], infer_datetime_format=True)
    print(df.type.unique())
    print(df.locale.unique())
    print(df.locale_name.unique())
    print(df.description.unique())
    print(df.transferred.unique())
    new_df = pd.DataFrame({"date": pd.date_range(start=min_date, end=max_date)})
    new_df = new_df.merge(df, left_on=["date"], right_on=["date"], how="left")
    new_df["year"] = new_df.date.dt.year
    new_df["month"] = new_df.date.dt.month
    new_df["day"] = new_df.date.dt.day
    new_df["wd"] = new_df.date.dt.dayofweek
    new_df["word_day"] = True
    new_df.loc[new_df.wd.isin([5, 6]), "word_day"] = False
    print(new_df.wd.unique())
    print(new_df.head(10))
    print(new_df.isna().sum())
    print(new_df.duplicated().sum())
    return new_df


def create_random_submission():
    if not exists(abspath("results/")):
        os.makedirs(abspath("results/"))
    df = pd.DataFrame({"id": np.arange(3000888, 3029400, dtype=np.int64),
                       "sales": 175 * np.random.random_sample(((3029399 - 3000888 + 1),))})
    print(f"Shape of the submit: {df.shape}")
    filename = join(abspath("results/"), f"sample_random_{dt.now().strftime('%Y-%m-%d_%H-%M')}.csv")
    df.to_csv(filename, index=False)
    return filename


def push_submission(file: str, message: str):
    if not isinstance(message, str):
        raise ValueError(f"Message must be str instead of {type(message)}.")
    if not exists(file):
        raise FileNotFoundError(f"No such {file}.")
    _, extension = splitext(file)
    if extension != ".csv":
        raise ValueError(f"File extension must be .csv not {extension}.")
    kaggle.api.competition_submit(competition="store-sales-time-series-forecasting", file_name=file, message=message)
    return kaggle.api.competition_submissions(competition="store-sales-time-series-forecasting")


def aggregate_data(train: pd.DataFrame, test: pd.DataFrame):
    transactions = load_transaction(train.date.min(), train.date.max(), train.store_nbr.unique())
    stores = load_stores()
    oils = load_oil(train.date.min(), train.date.max())
    holidays = load_holidays(train.date.min(), train.date.max())
    # print(stores.city.unique())
    # print(len(stores.city.unique()))
    # print(holidays.locale_name.unique())
    # print(len(holidays.locale_name.unique()))
    # print(set(holidays.locale_name).difference(set(stores.city)))
    # replacer_city = {'Santo Domingo de los Tsachilas': 'Santo Domingo'}
    new_train, new_test = train.copy(), test.copy()
    new_train = new_train.merge(stores, left_on=["store_nbr"], right_on=["store_nbr"], how="left")
    new_train = new_train.merge(oils, left_on=["date"], right_on=["date"], how="left")
    new_train = new_train.merge(transactions, left_on=["date", "store_nbr"], right_on=["date", "store_nbr"], how="left")
    # new_train = new_train.merge(holidays, left_on=["date", "city"], right_on=["date", "locale_name"], how="left")
    return preprocessing(new_train), new_test


def preprocessing(df: pd.DataFrame):
    new_df = df.copy()
    new_df["date"] = new_df.date.apply(lambda x: x.toordinal())
    date_scaler = MinMaxScaler()
    date_scaler.fit(new_df.date.values.reshape(-1, 1))
    new_df["date"] = date_scaler.transform(new_df.date.values.reshape(-1, 1))
    store_nbr_encoder = LabelEncoder()
    store_nbr_encoder.fit(new_df.store_nbr)
    new_df["store_nbr"] = store_nbr_encoder.transform(new_df.store_nbr)
    family_encoder = LabelEncoder()
    family_encoder.fit(new_df.family)
    new_df["family"] = family_encoder.transform(new_df.family)
    onpromotion_scaler = MinMaxScaler()
    onpromotion_scaler.fit(new_df.onpromotion.values.reshape(-1, 1))
    new_df["onpromotion"] = onpromotion_scaler.transform(new_df.onpromotion.values.reshape(-1, 1))
    city_encoder = LabelEncoder()
    city_encoder.fit(new_df.city)
    new_df["city"] = city_encoder.transform(new_df.city)
    state_encoder = LabelEncoder()
    state_encoder.fit(new_df.state)
    new_df["state"] = state_encoder.transform(new_df.state)
    type_encoder = LabelEncoder()
    type_encoder.fit(new_df.type)
    new_df["type"] = type_encoder.transform(new_df.type)
    cluster_encoder = LabelEncoder()
    cluster_encoder.fit(new_df.cluster)
    new_df["cluster"] = cluster_encoder.transform(new_df.cluster)
    oil_scaler = MinMaxScaler()
    oil_scaler.fit(new_df.dcoilwtico.values.reshape(-1, 1))
    new_df["dcoilwtico"] = oil_scaler.transform(new_df.dcoilwtico.values.reshape(-1, 1))
    transactions_scaler = MinMaxScaler()
    transactions_scaler.fit(new_df.transactions.values.reshape(-1, 1))
    new_df["transactions"] = transactions_scaler.transform(new_df.transactions.values.reshape(-1, 1))
    sales_scaler = MinMaxScaler()
    sales_scaler.fit(new_df.sales.values.reshape(-1, 1))
    new_df["sales"] = sales_scaler.transform(new_df.sales.values.reshape(-1, 1))
    return new_df


# Main thread
if __name__ == "__main__":
    start = dt.now()
    print(f"Starting script at {start.strftime('%Y-%m-%d %H:%M')}")
    train = load_train()
    test = load_test()
    train, test = aggregate_data(train, test)
    print(train.head())
    print(train.tail())
    # print(train.isna().sum())
    print(push_submission(create_random_submission(), "175 random"))
    print(f"It takes {dt.now() - start}.")
