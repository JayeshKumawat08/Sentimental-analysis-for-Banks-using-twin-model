import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Final_DeepLearning_Dataset.csv")

if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(BASE_DIR, "..", "Final_DeepLearning_Dataset.csv")

print(f"Loading dataset from: {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

df.dropna(subset=['Cleaned_Content'], inplace=True)
print("Vectorizing Hinglish text data...")
vectorizer = TfidfVectorizer(max_features=5000) 
X = vectorizer.fit_transform(df['Cleaned_Content'])
y = df['score'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Linear Regression using SGD...")
sgd_model = SGDRegressor(
    loss='squared_error',      
    learning_rate='constant',  
    eta0=0.01,                 
    random_state=42
)

epochs = 100
loss_history = []

print("Training Linear Regression using SGD and tracking loss...")
for epoch in range(epochs):
    
    sgd_model.partial_fit(X_train, y_train)
    
    train_predictions = sgd_model.predict(X_train)
    current_mse = mean_squared_error(y_train, train_predictions)
    loss_history.append(current_mse)

predictions = sgd_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Evaluation ---")
print(f"MSE: {mse:.4f} (Lower is better)")
print(f"RMSE: {rmse:.4f} (Lower is better)")
print(f"MAE: {mae:.4f} (Lower is better)")
print(f"R2 Score: {r2:.4f} (Closer to 1 is better)")


plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), loss_history,
         color='red', linewidth=2,
         label='Training Loss (MSE)')
plt.title("Loss vs Epochs", fontsize=13)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

window = 5
smoothed_loss = np.convolve(
    loss_history,
    np.ones(window) / window,
    mode='valid'
)

plt.subplot(1, 2, 2)
plt.plot(range(window, epochs + 1),
         smoothed_loss,
         color='blue',
         linewidth=2,
         label='Convergence Curve')

convergence_epoch = np.argmin(smoothed_loss) + window
plt.axvline(convergence_epoch,
            color='green',
            linestyle='--',
            label=f'Converged ~ Epoch {convergence_epoch}')

plt.title("Convergence (Loss Curve)", fontsize=13)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()