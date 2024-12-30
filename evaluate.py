import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_prediction(true_price, pred_price):
    mae = mean_absolute_error(true_price, pred_price)
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

def plot_result(true_price, pred_price):
    plt.figure(figsize=(10, 6))
    plt.plot(true_price, label='True', color='blue')
    plt.plot(pred_price, label='Predicted', color='orange')
    plt.legend()
    plt.title('Prediction vs. True Price')
    plt.show()
