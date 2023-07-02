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

fig = px.box(df_renamed,
             y="WindSpeed (m/s)",
             title="Общая статистика скорости ветра")

fig.show()

fig = px.box(df_renamed,
             y="Humidity (%)",
             title="Общая статистика влажности")

fig.show()

df_corr = df_renamed.corr()
df_corr

x = list(df_corr.columns)
y = list(df_corr.index)
z = np.array(df_corr)

fig = ff.create_annotated_heatmap(x = x,
                                  y = y,
                                  z = z,
                                  annotation_text = np.around(z, decimals=2))
fig.show()

fig = px.scatter(df_renamed,
                 x="PowerConsumption (W)",
                 y="Temperature (°C)",
                 title = "Потребление энергии vs Температура")
fig.show()

fig = px.scatter(df_renamed,
                 x="PowerConsumption (W)",
                 y="WindSpeed (m/s)",
                 title = "Потребление энергии vs Скорость ветра")
fig.show()

fig = px.scatter(df_renamed,
                 x="PowerConsumption (W)",
                 y="Humidity (%)",
                 title = "Потребление энергии vs Влажность")
fig.show()

df_renamed.plot.scatter(x='Temperature (°C)',y='PowerConsumption (W)' )

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df_renamed['PowerConsumption (W)'])
plt.title('Распределение потребления', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df_renamed['Temperature (°C)'])
plt.title('Распределение температуры', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df_renamed['Humidity (%)'])
plt.title('Распределение влажности', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

fig, ax = plt.subplots(figsize = (20,15))
plt.hist(df_renamed['WindSpeed (m/s)'])
plt.title('Распределение скорости ветра', fontsize = 24)
ax.set_xlabel('Все что угодно', fontsize = 24)
ax.set_ylabel('y' , fontsize = 24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()

df_renamed.info()


