import numpy as np
import matplotlib.pyplot as plt

# --- データ ---
distances = np.array([
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00
])

pixel_diams = np.array([
    657.70, 400.12, 278.09, 217.17, 180.26, 153.56, 132.53, 116.98,
    104.10, 93.64, 86.22, 79.87, 73.14, 67.84, 63.43, 59.58,
    55.59, 52.74, 49.91, 47.12
])

# --- 近似計算 ---
inv_pixel = 1 / pixel_diams
a, b = np.polyfit(inv_pixel, distances, 1)
estimated = a / pixel_diams + b
errors = estimated - distances
abs_errors = np.abs(errors)
mae = np.mean(abs_errors)

# --- プロット ---
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# (1) 上段：距離 vs ピクセル直径
axs[0].scatter(pixel_diams, distances, label="測定データ", color="blue", s=50)
axs[0].plot(pixel_diams, estimated, label=f"近似式: D = {a:.2f} / px + {b:.2f}", color="red")
axs[0].invert_xaxis()
axs[0].set_xlabel("Pixel Diameter [px]")
axs[0].set_ylabel("Distance [m]")
axs[0].set_title("距離とピクセル直径の関係")
axs[0].legend()
axs[0].grid(True)

# (2) 下段：誤差グラフ
axs[1].bar(distances, abs_errors, width=0.03, color='orange', edgecolor='black')
axs[1].set_xlabel("Actual Distance [m]")
axs[1].set_ylabel("Absolute Error [m]")
axs[1].set_title(f"距離推定の誤差（平均誤差: {mae:.4f} m）")
axs[1].grid(True)

plt.tight_layout()
plt.show()

