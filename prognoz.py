#Импорт библиотек
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

# Разделить датасет на два датафрейма
df = pd.read_csv("powerconsumption.csv")
df['Datetime'] = pd.to_datetime(df.Datetime, format='%m/%d/%Y %H:%M')

#Создание столбца, в котором находится среднее значение из 3-х зон
df['PowerConsumption'] = (df[['PowerConsumption_Zone1','PowerConsumption_Zone2','PowerConsumption_Zone3']].sum(axis=1))/3

#Удаление лишних столбцов
df. drop (columns=df. columns [8], axis= 1 , inplace= True )
df.drop (columns=df.columns [7], axis= 1 , inplace= True )
df.drop (columns=df.columns [6], axis= 1 , inplace= True )

#Создаем столбец Month
df["Month"] = df["Datetime"].dt.month
df_1_to_11 = df[df['Month'].isin(range(1, 12))] #В переменную записываем месяцы с 1 по 11
df_12 = df[df['Month'] == 12] #В переменную записываем 12 месяц

#Вывести первые 5 рядов каждого датафрейма
df_1_to_11.to_csv("1to11.csv", index=False)
df_12.to_csv("to12.csv", index=False)
print(df_1_to_11.head())
print(df_12.head())

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

df_1_to_11 = df_1_to_11.drop('Datetime', axis = 1)

# Разделим датасет на обучающую и тестовую выборку
train_df, test_df = train_test_split(df_1_to_11, test_size=0.2, random_state=42)

# Поменяем обучающую и тестовую data информацию в 3D массив для ввода LSTM
X_train = train_df.drop("PowerConsumption", axis=1).values.reshape(-1, 1, 6)
X_test = test_df.drop("PowerConsumption", axis=1).values.reshape(-1, 1, 6)
y_train = train_df["PowerConsumption"].values.reshape(-1, 1)
y_test = test_df["PowerConsumption"].values.reshape(-1, 1)