# Energy consumption of the Tétouan city in Morocco

#Импорт библиотек
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

#Загрузка файла
df = pd.read_csv("powerconsumption.csv")

#Отобразить верхнюю часть данных
df.head()

#Отобразить нижнюю часть данных
df.tail()

#Отобразить тип данных и проверить на нулевые значения
df.info()

#Отобразить состав набора данных
df.shape

#Проверить на пустые значения
df.isnull().sum()

#Проверка на наличие дубликатов в данных
df.duplicated().sum()

#Просмотр описательной статистики в наборе данных
df.describe()

#Создание столбца, в котором находится среднее значение из 3-х зон
df['PowerConsumption'] = (df[['PowerConsumption_Zone1','PowerConsumption_Zone2','PowerConsumption_Zone3']].sum(axis=1))/3

#Удаление лишних столбцов
df. drop (columns=df. columns [8], axis= 1 , inplace= True )
df.drop (columns=df.columns [7], axis= 1 , inplace= True )
df.drop (columns=df.columns [6], axis= 1 , inplace= True )

df_renamed=df.rename(columns={'Temperature':'Temperature (°C)', 'Humidity':'Humidity (%)', 'WindSpeed':'WindSpeed (m/s)', 'GeneralDiffuseFlows':'GeneralDiffuseFlows','PowerConsumption':'PowerConsumption (W)'})

df_renamed['Datetime'] = pd.to_datetime(df.Datetime, format='%m/%d/%Y %H:%M')

#Создание нового столбца в наборе данных (Месяцы)
df_renamed["Month"] = df_renamed["Datetime"].dt.month
df_renamed

#Установка индекса на столбец Datetime
df_renamed = df_renamed.set_index(df["Datetime"])
df_renamed

#Группировка данных по месяцам
grouped = df_renamed.groupby('Month').mean(numeric_only=True)
grouped

fig = px.line(df_renamed,
              x="Datetime",
              y='PowerConsumption (W)',
              labels = {'Datetime':'Months of the year'},
              title = "Потребление электроэнергии с января по декабрь")
fig.update_layout(
    template='plotly',
    font=dict(size=10),
    title={
        'text': "Потребление электроэнергии с января по декабрь",
        'font': {'size': 34}
    }
)
fig.show()

fig = px.box(df_renamed,
        x=df_renamed["Datetime"].dt.month,
        y="PowerConsumption (W)",
        color=df_renamed["Datetime"].dt.month,
        labels = {"x" : "Месяцы"},
        title="Месячная статистика потребления электроэнергии ")

fig.update_traces(width=0.5)
fig.show()

fig = px.box(df_renamed,
        x=df_renamed["Datetime"].dt.day,
        y="PowerConsumption (W)",
        color=df_renamed["Datetime"].dt.day,
        labels = {"x" : "Дни"})

fig.update_traces(width=0.5)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="PowerConsumption (W)",
              labels = {'Month':'Месяцы'},
              color = "PowerConsumption (W)",
              title="Потребление электроэнергии в месяц")
fig.update_traces(width=0.8)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Temperature (°C)",
              labels = {'Month':'Месяцы'},
              color = "Temperature (°C)",
              title="Температура в месяце")
fig.update_traces(width=0.8)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="Humidity (%)",
              labels = {'Month':'Месяцы'},
              color = "Humidity (%)",
              title="Влажность в месяце")
fig.update_traces(width=0.8)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.bar(grouped,
              x=grouped.index,
              y="WindSpeed (m/s)",
              labels = {'Month':'Месяцы'},
              color = "WindSpeed (m/s)",
              title="Скорость ветра в месяце")
fig.update_traces(width=0.8)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

fig = px.box(df_renamed,
             y="PowerConsumption (W)",
             title="Общая статистика потребления электроэнергии")

fig.show()

fig = px.box(df_renamed,
             y="Temperature (°C)",
             title="Общая статистика температуры")

fig.show()

