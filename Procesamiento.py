import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generar señal de audio sintética con ruido
fs = 8000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs)
signal_clean = np.sin(2 * np.pi * 440 * t)  # Señal senoidal de 440 Hz
noise = np.random.normal(0, 0.3, signal_clean.shape)
signal_noisy = signal_clean + noise

# Preparar datos para la red neuronal
X = torch.tensor(signal_noisy, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(signal_clean, dtype=torch.float32).unsqueeze(1)

# Definir modelo simple (MLP)
class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = DenoiseNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenamiento
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Resultado
with torch.no_grad():
    denoised = model(X).squeeze().numpy()

# Graficar
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(t, signal_clean)
plt.title("Señal limpia (original)")
plt.subplot(3,1,2)
plt.plot(t, signal_noisy)
plt.title("Señal con ruido")
plt.subplot(3,1,3)
plt.plot(t, denoised)
plt.title("Señal filtrada con IA")
plt.tight_layout()
plt.show()
