import pandas as pd
import numpy as np
from pandas import concat


df = pd.read_excel("data_ok.xlsx", sep='delimiter', header=0)
df['DATEPRD'] = pd.to_datetime(df["DATEPRD"]).dt.date
print(df.head())
print(df.keys())

out_col = df.loc[:, ['BORE_OIL_VOL']]
df = df.drop(["BORE_OIL_VOL"], axis=1)
df = df.join(out_col)

def series_to_supervised(dataframe, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    	data: Sequence of observations as a list or NumPy array.
    	n_in: Number of lag observations as input (X).
    	n_out: Number of observations as output (y).
    	dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    	Pandas DataFrame of series framed for supervised learning.
    """
    n_vars =1 if type(dataframe) is list else dataframe.shape[1]
    cols, names = list(), list()
    #input sequence(t-n,..t-1)
    for i in range(n_in, 0, -1):
        cols.append(dataframe.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars-1)]
        names += [('Output(t-%d)' % i)]

    #forcast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(dataframe.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars-1)]
            names += ['Output(t)']
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            names += ['Output(t)']

    #put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    #drop rows with Nan
    if dropnan:
        agg.dropna(inplace=True)
    return agg


data = series_to_supervised(df)
print(data.head())