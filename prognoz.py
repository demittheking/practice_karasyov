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