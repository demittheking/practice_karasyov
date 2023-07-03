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
