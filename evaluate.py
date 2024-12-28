import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_prediction(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("----------------------------------------")
    print("Evaluation Metrics")
    print("----------------------------------------")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-Squared (R²): {r2:.4f}")
    print("----------------------------------------")

def plot_result(y_true, y_pred):
    import numpy as np
    import matplotlib.pyplot as plt

    y_true_1d = y_true.flatten()
    y_pred_1d = y_pred.flatten()

    plt.figure(figsize=(10,6))
    plt.plot(y_true_1d, label="Actual")
    plt.plot(y_pred_1d, label="Predicted")
    plt.fill_between(range(len(y_true_1d)), y_true_1d, y_pred_1d, color='gray', alpha=0.2, label='Error Area')
    plt.legend()
    plt.show()
