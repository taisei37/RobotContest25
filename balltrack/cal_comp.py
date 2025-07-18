import cv2
import numpy as np
import matplotlib.pyplot as plt

# === キャリブレーション済みパラメータ ===
camera_matrix = np.array([[750.10059546, 0., 704.54913907],
                          [0., 746.54075486, 445.7714058],
                          [0., 0., 1.]])

dist_coeffs = np.array([0.04739503, -0.07422041, 0.00880341, 0.0123376, 0.02295108])

# === グリッド描画関数 ===
def draw_grid(img, grid_size=50, color=(0, 255, 0), thickness=1):
    h, w = img.shape[:2]
    for x in range(0, w, grid_size):
        cv2.line(img, (x, 0), (x, h), color, thickness)
    for y in range(0, h, grid_size):
        cv2.line(img, (0, y), (w, y), color, thickness)
    return img

# === カメラ起動 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 675)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("❌ カメラが開けませんでした")
    exit()

print("✅ カメラ準備完了。スペースキーで写真を撮影、ESCで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 映像取得に失敗しました")
        break

    cv2.imshow("Live View (Press Space to Capture)", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        print("🔚 終了します")
        break

    elif key == 32:  # SPACE
        print("📸 撮影中...")

        # === 歪み補正 ===
        h, w = frame.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        # === グリッド描画（コピーして別画像に描画）===
        original_with_grid = draw_grid(frame.copy(), grid_size=75)
        undistorted_with_grid = draw_grid(undistorted.copy(), grid_size=75)

        # === 比較表示 ===
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Before Calibration (Grid Overlay)")
        plt.imshow(cv2.cvtColor(original_with_grid, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("After Calibration (Grid Overlay)")
        plt.imshow(cv2.cvtColor(undistorted_with_grid, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()

cap.release()
cv2.destroyAllWindows()

