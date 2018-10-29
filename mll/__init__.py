import math

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


def parse_date(df, field_name, drop_origin=False):
    field = df[field_name]
    if not np.issubdtype(field, np.datetime64):
        return

    subs = ['year', 'month', 'week', 'day',
            'dayofweek', 'dayofyear', 'is_month_end',
            'is_month_start', 'is_quarter_end',
            'is_quarter_start', 'is_year_end', 'is_year_start',
            'hour', 'minute', 'second']

    for sub in subs:
        df['{}_{}'.format(field_name, sub)] = getattr(field.dt, sub.lower())

    # Convert nanosec to sec
    df[field_name + '_elapsed'] = field.astype(np.int64) // 10 ** 9

    if drop_origin:
        df.drop(field_name, axis=1, inplace=True)


def str_to_category(df):
    for k, v in df.items():
        if pd.api.types.is_string_dtype(v):
            df[k] = v.astype('category').cat.as_ordered()


def fill_missing(df):
    for name, field in df.items():
        if not pd.api.types.is_numeric_dtype(field):
            continue

        df[name + '_na'] = pd.isnull(field)
        df[name] = field.fillna(field.median())


def category_to_code(df):
    for name, field in df.items():
        if pd.api.types.is_categorical_dtype(field):
            # The missing value of category is -1, increase it to 0.
            df[name] = field.cat.codes + 1


def process(df, y_name, sample_size=None):
    if sample_size is None:
        df_copy = df.copy()
    else:
        df_copy = sample(df, sample_size)

    y = df_copy[y_name].values
    df_copy.drop(y_name, axis=1, inplace=True)

    fill_missing(df_copy)
    category_to_code(df_copy)
    return df_copy, y


def split(df, number):
    return df[:number].copy(), df[number:].copy()


def rmse(predict, actual):
    return math.sqrt(((predict - actual) ** 2).mean())


def scores(m, X_train, y_train, X_valid, y_valid):
    return [
        'train rmse: {:f}'.format(rmse(m.predict(X_train), y_train)),
        'train r^2: {:f}'.format(m.score(X_train, y_train)),
        'validate rmse: {:f}'.format(rmse(m.predict(X_valid), y_valid)),
        'validate r^2: {:f}'.format(m.score(X_valid, y_valid))
    ]


def sample(df, sample_size):
    indexes = sorted(np.random.permutation(len(df))[:sample_size])
    return df.iloc[indexes].copy()


def parallel(func, iterables, job_num=8):
    return list(ProcessPoolExecutor(job_num).map(func, iterables))
