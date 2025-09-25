import pandas as pd

data = pd.read_csv('../Microestructura_y_Sistemas_de_Trading/data/aapl_5m_train.csv').dropna()
print(data.head())