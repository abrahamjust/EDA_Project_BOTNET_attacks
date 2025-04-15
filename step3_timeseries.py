import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load the dataset
df = pd.read_csv("combined_data.csv")

# Step 2: Split the data
benign_data = df[df['attack_type'] == 'benign'].reset_index(drop=True)
mirai_data = df[df['attack_type'].str.contains('mirai', case=False)].reset_index(drop=True)
gafgyt_data = df[df['attack_type'].str.contains('gafgyt', case=False)].reset_index(drop=True)

# Step 3: Define the best feature
feature_column = 'HH_L0.01_mean'  # This was found to be informative

def apply_arima(series, label, steps=10, clip_quantile=0.99, last_n=1000):
    """
    Applies ARIMA forecasting to the last `last_n` points of the series,
    clips values for better visualization, and plots both original and forecasted data.
    """
    # Select last N points and reset index for clean x-axis
    subset = series[-last_n:].reset_index(drop=True)

    # Fit ARIMA model
    model = ARIMA(subset, order=(5,1,0))  # Order can be tuned
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    # Print debug info
    print(f"\n{label} Data:")
    print("Max:", subset.max(), "Min:", subset.min())
    print("Forecast:", forecast.tolist())

    # Optional clipping to reduce noise from outliers
    clipped = subset.clip(upper=subset.quantile(clip_quantile))  # Clip top 1% outliers

    # Local x-axis for plotting
    x_original = list(range(len(clipped)))
    x_forecast = list(range(len(clipped), len(clipped) + steps))

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(x_original, clipped, label='Original Data (Clipped)', color='blue')
    plt.plot(x_forecast, forecast, label='Forecasted Data', color='red')
    plt.title(f'{label} Data Forecast using ARIMA (Last {last_n} Points)')
    plt.xlabel("Time Index")
    plt.ylabel("Feature Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Apply ARIMA to each category
apply_arima(benign_data[feature_column], "Benign")
apply_arima(mirai_data[feature_column], "Mirai")
apply_arima(gafgyt_data[feature_column], "Gafgyt")
