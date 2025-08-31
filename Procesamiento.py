import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Cargar audio (mono)
fs, signal_clean = wavfile.read("entrada.wav")
signal_clean = signal_clean.astype(np.float32)
signal_clean = signal_clean / np.max(np.abs(signal_clean))  # normalizar [-1,1]

# Crear versión con ruido
noise = np.random.normal(0, 0.1, signal_clean.shape)
signal_noisy = signal_clean + noise

# Convertir a tensores
X = torch.tensor(signal_noisy, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(signal_clean, dtype=torch.float32).unsqueeze(1)

# Modelo simple
class DenoiseNet(nn.Module):
    def __init__(self):
        super(DenoiseNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = DenoiseNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entrenar
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# Resultado
with torch.no_grad():
    denoised = model(X).squeeze().numpy()

# Guardar audio filtrado
wavfile.write("salida_filtrada.wav", fs, (denoised * 32767).astype(np.int16))

# Graficar
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.plot(signal_clean)
plt.title("Señal limpia (original)")
plt.subplot(3,1,2)
plt.plot(signal_noisy)
plt.title("Señal con ruido")
plt.subplot(3,1,3)
plt.plot(denoised)
plt.title("Señal filtrada con IA")
plt.tight_layout()
plt.show()
