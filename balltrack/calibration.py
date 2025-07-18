import cv2
import numpy as np
from matplotlib import pyplot as plt

CHECKERBOARD =( 7,7 )
SQUARE_SIZE = 25

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# === カメラ起動チェック付き設定 ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 675)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("❌ カメラを開けませんでした。デバイス番号（VideoCaptureの引数）を確認してください。")
    exit()

print("✅ カメラ接続成功。スペースキーでキャプチャ、ESCキーで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ フレームを取得できませんでした。")
        break

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if found:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, found)

    cv2.putText(display, f"Captures: {len(objpoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Camera View", display)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32 and found:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        objpoints.append(objp)
        print(f"> 📸 キャプチャ成功：{len(objpoints)} 枚目")

cap.release()
cv2.destroyAllWindows()

# === キャリブレーション実行 ===
if len(objpoints) >= 5:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n=== 🎯 キャリブレーション結果 ===")
    print("カメラ行列（Camera Matrix）:")
    print(camera_matrix)

    print("\n歪み係数（Distortion Coefficients）:")
    print(dist_coeffs.ravel())

    print("\n回転ベクトル（Rotation Vector）:")
    print(rvecs[0].ravel())

    print("\n並進ベクトル（Translation Vector）:")
    print(tvecs[0].ravel())

    # 歪み補正表示
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title("Undistorted")
    plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
    plt.show()
else:
    print("⚠ キャリブレーションには少なくとも5枚のキャプチャが必要です。")

