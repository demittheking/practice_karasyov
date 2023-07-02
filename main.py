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

