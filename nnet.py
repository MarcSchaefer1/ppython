import pandas as pd
import requests
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# 1. Koordinaten einlesen
stations = pd.read_csv(
    "weather.csv",
    encoding="utf-8-sig"
)

# 2. Alle 21 Standorte mit einem API-Request abfragen
params = {
    "latitude": ",".join(stations["Lat"].astype(str)),
    "longitude": ",".join(stations["Lon"].astype(str)),
    "start_date": "2011-01-01",
    "end_date": "2025-12-31",
    "daily": "temperature_2m_max",
    "timezone": "Europe/Berlin"
}

response = requests.get(
    "https://archive-api.open-meteo.com/v1/archive",
    params=params,
    timeout=60
)

response.raise_for_status()
weather_data = response.json()

print(type(weather_data))

# 3. Temperaturwerte in einen Tensor umwandeln
# Ursprüngliche Form: [21 Stationen, Anzahl Tage]
temperatures = torch.tensor(
    [
        location["daily"]["temperature_2m_max"]
        for location in weather_data
    ],
    dtype=torch.float32
)

# Gewünschte Form: [Anzahl Tage, 21 Stationen]
temperatures = temperatures.T

# 4. Input und Target erzeugen
X = temperatures[:-1, 1:]   # Stationen 2–21 am aktuellen Tag [day, station]
y = temperatures[1:, 0:1]   # Station 1 am nächsten Tag

print("Target-Station:", stations.iloc[0]["Name"])
print("X:", X.shape)
print("y:", y.shape)

# 5. Dataset und DataLoader erzeugen
dataset = TensorDataset(X, y)

data_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Einen Batch ansehen
X_batch, y_batch = next(iter(data_loader))

print("Input-Batch:", X_batch.shape)
print("Target-Batch:", y_batch.shape)

class NNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NNet().to(torch.device("cpu"))
print(model)
epochen = 100

def loss_fn(predictions, targets):
    return torch.mse(predictions, targets)

def optimizer_fn(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

optimizer = optimizer_fn(model)

model.train()

def train_model(model, data_loader, loss_fn, optimizer, epochen):
    for epoch in range(epochen):
        for X_batch, y_batch in data_loader:
            # Vorwärtsdurchlauf
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)

            # Rückwärtsdurchlauf und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochen}, Loss: {loss.item():.4f}")

train_model(model, data_loader, loss_fn, optimizer, epochen)