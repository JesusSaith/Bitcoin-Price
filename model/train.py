from joblib import dump

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pathlib  
from os import PathLike
from sklearn.metrics import classification_report

def load_model(model_path: PathLike) -> RandomForestRegressor:
  """Loads a random forest regressor model from a file."""
  import joblib
  with open(model_path, "rb") as f:
    model = joblib.load(f)
  return model

def predict_bitcoin_price(model: RandomForestRegressor, X: pd.DataFrame) -> float:
  """Predicts the price of Bitcoin using a random forest regressor model."""
  price_prediction = model.predict(X)[0]
  return price_prediction

# Cargar los datos
df = pd.read_csv("data/main.csv")

# Preparar los datos de entrada
X = df[['Open', 'Close', 'Volume']]

# Definir los datos de salida
y = df['Close']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar el modelo
clf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
clf.fit(X_train, y_train)

# Guardar el modelo entrenado
dump(clf, "model/bitcoin-v1.joblib")

# Actualizar el modelo
def train_model(model: RandomForestRegressor):
  """Trains a random forest regressor model."""

  # Establecer los parámetros del modelo
  model.set_params(n_estimators=20)

  # Guardar el modelo actualizado
  dump(model, "model/bitcoin-v4.joblib")

if __name__ == "__main__":
  # Entrenar el modelo
  clf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=0)
  clf.fit(X_train, y_train)

  # Actualizar el modelo
  train_model(clf)