import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from urllib.request import urlretrieve


def plot_model_on_data(x, y, model=None, title=None, ax=None):
    if ax is None:
        plt.figure(figsize=(9,6))
        ax = plt.gca()
    ax.scatter(x, y)
    if model is not None:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        plot_x = np.linspace(xlim[0], xlim[1], 100)
        plot_y = model(plot_x)
        ax.plot(plot_x, plot_y, lw=3, c="red")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("Picco consumi (GW)")
    
# Uniform Linear Regression _ Mean Squared Error
def ulr_mse(x, y, alpha, beta):
    return np.mean(np.square(alpha * x + beta - y))

# 0.2 and 3 are arbitrary values

def make_model(alpha, beta):
    def model(x):
        return alpha * x + beta
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
    
    # print(sample_model(np.array([20, 25, 30])))
    
    sample_model = make_model(0.2, -3)
    
    # print(sample_model(np.array([20, 25, 30])))
    
    sample_model_vec = np.vectorize(sample_model)
    
    # print(sample_model_vec(25))

    # print(sample_model_vec([20, 25, 30]))
    
    plot_x = np.linspace(15, 35, 40)
    # print(plot_x)
    
    plot_y = sample_model(plot_x)
    
    # print(plot_y)
    
    # plt.figure(figsize=(9, 6))
    # plt.plot(plot_x, plot_y)

    # # aggiungo linee guida lungo entrambi gli assi
    # plt.grid()

    # # assegno etichette comprensibili agli assi
    # plt.xlabel("Temperatura (°C)")
    # plt.ylabel("Picco consumi (GW)")    
    
    # returns Axes as an object apparently allows us to create multiple graphs in one
    # ax = plt.gca()

    # ax.scatter(temp, demand)
    
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()
    
    # # lw = line width
    # ax.plot(plot_x, plot_y, lw=3, c="red")
    
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    
    # ax.grid()
    # ax.set_xlabel("Temperatura (°C)")
    # ax.set_ylabel("Picco consumi (GW)")
    
    # plot_model_on_data(temp, demand, sample_model, "Test Model")
    
    predicted_demand = sample_model(temp)
    
    error = predicted_demand - demand
    
    print(np.mean(np.square(error)))
    
    new_model = make_model(0.15, -1)
    
    print(new_model(np.array([20, 25, 30])))
    
    
    plot_model_on_data(temp, demand, new_model, "New Model")
    
    new_predict_demand = new_model(temp)
    
    error = new_predict_demand - demand
    
    print("New Error => ", np.mean(np.square(error)))
    
    plt.show()
    

if __name__ == "__main__":
     main()
