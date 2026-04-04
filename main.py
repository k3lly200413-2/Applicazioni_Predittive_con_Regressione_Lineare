import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from urllib.request import urlretrieve

# 0.2 and 3 are arbitrary values 

def make_model(alpha, beta):
    def model(x):
        return alpha * beta - 3
    return model

def setup():
    
    DATA_URL = "https://github.com/datascienceunibo/dialab2024/raw/main/Regressione_Lineare/power.csv"

    if not os.path.exists("power.csv"):
        urlretrieve(DATA_URL, "power.csv")


def main():
    setup()
    
    data = pd.read_csv("power.csv", parse_dates=["date"], index_col="date")
    
    # print(data.tail(3))
    
    # print(data.index.month)

    # print((data.index.month >= 6) & (data.index.month <= 8))
    
    # print(data.loc[(data.index.month >= 6) & (data.index.month <= 8)])
    
    data_summer = data.loc[data.index.month.isin([6, 7, 8])]
    
    # print(data_summer.head(3))
    # print(data_summer.tail(3))
    
    # data_summer["temp"].plot.hist(bins=20)
    # data_summer["demand"].plot.hist(bins=20)
    
    # data_summer.plot.scatter("temp", "demand")
    
    # plt.show()
    
    # print(data["temp"].describe())
    # print(data["demand"].describe())

    temp = data_summer["temp"].values
    demand = data_summer["demand"].values
    
    # print(np.mean((temp - temp.mean()) * (demand - demand.mean())) / (temp.std() * temp.std()))
    
    # same as
    # Transposed because we need the values to be on thw row and not on the column
    
    # print(np.corrcoef(data_summer.T))
    
    print(sample_model(np.array([20, 25, 30])))

if __name__ == "__main__":
     main()
