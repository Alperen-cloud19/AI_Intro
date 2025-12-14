import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Datas = np.array([
    [0.75, 109.1],  # Parlak ve Ağır -> Metal (1)
    [0.98, 35.9],   # Parlak ama Hafif (Folyo vb.) -> Çöp (0)
    [0.839, 11.1],  # Çok Hafif -> Çöp (0)
    [0.889, 105.2], # İdeal Metal -> Metal (1)
    [0.662, 224.3], # Yoğun ve Ağır -> Metal (1)
    [0.623, 134.6], # Standart Metal -> Metal (1)
    [0.946, 128.8], # Yüksek Kalite -> Metal (1)
    [0.256, 45.7],  # Mat ve Hafif (Plastik) -> Çöp (0)
    [0.123, 123.4], # Ağır ama Mat (Taş/Odun) -> Çöp (0)
    [0.334, 132.5], # Yeterince Parlak Değil -> Çöp (0)
    [0.781, 67.7]   # Sınırda (Hafif Kaldı) -> Çöp (0)
])
y = np.array([1,0,0,1,1,1,1,0,0,0,0])

# For graph
Datas_raw = Datas.copy()

# Normalization
max_vec = np.max(Datas, axis=0)
Datas = Datas / max_vec

# ------- Perception Parameters -------
np.random.seed(42) # Sonuçlar tutarlı olsun diye
weight = np.random.rand(2) * 0.1
bias = 0
lr = 0.1

def activation(x):
    return 1 if x >= 0 else 0

# ------- Training Loop --------
for epoch in range(50):
    total_error = 0
    for xi, target in zip(Datas, y):
        x = np.dot(xi, weight) + bias
        pred = activation(x)
        error = target -pred

        # mass configuration
        weight += lr * error * xi
        bias += lr * error

        total_error += abs(error)

    if total_error == 0:
        break
print("\nSon Ağırlıklar:", weight)
print("Bias:", bias)
print("Eğitim Tamamlandı")

# --------- Test Section  ----------
# We need to store the normalization coefficient
# Note: In your code, the Datas variable changes after training,
# so we find the max value again from the raw data, Datas_raw.
max_val = np.max(Datas_raw)
def predict(shine, weight_val):
    # 1. Convert the data received from the user into a vector.
    vec = np.array([shine, weight_val])
    # 2. Apply the same normalization (This is the critical point)
    vec = vec / max_vec
    #3. Calculation (Weight * Input + Bias)
    z = np.dot(vec, weight) + bias
    # 4. Put it into the activation function.
    return activation(z)

print("\n--- Örnek Test ---")
try:
    # Kullanıcıdan veri al
    input_shine = float(input("Parlaklık (Metallic Shine) değerini girin (örn: 0.85): "))
    input_weight = float(input("Ağırlık (Weight) değerini girin (örn: 120.5): "))
    # Tahmin yap
    sonuc = predict(input_shine, input_weight)

    if sonuc == 1:
        print(f"SONUÇ: Geri Dönüştürülebilir (Recyclable) - [Metal]")
    else:
        print(f"SONUÇ: Geri Dönüştürülemez (Unrecyclable)")

except ValueError:
    print("Lütfen geçerli sayısal değerler giriniz.")