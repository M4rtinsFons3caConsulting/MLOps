"""
This is a pipeline 'data_splitting'
generated using Kedro 0.19.14
"""

import pandas as pd

from sklearn.model_selection import train_test_split


def split_data(
    data: pd.DataFrame
    ,test_size=0.2
    ,random_state=20
):
    train, test = train_test_split(
        data
        ,test_size=test_size
        ,random_state=random_state
        ,shuffle=False  # working with time series
    )

    return train.reset_index(drop=True), test.reset_index(drop=True)