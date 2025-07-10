import cv2
import numpy as np
import time

# --- ユーザー設定部分 ---
DEVICE = '/dev/video4'
KNOWN_DISTANCE = 6.3   # ボールまでの既知の距離 [cm]
BALL_DIAMETER = 6.5     # ボールの実直径 [cm]
SAMPLES = 30            # サンプリング数
CALIB_COLOR = 'red'     # キャリブレーションに使う色（'red','blue','yellow'）

# 赤色用HSV範囲（必要に応じて調整）
color_ranges = {
    "red":   (np.array([165, 105, 115]), np.array([175, 250, 255])),
    "blue":  (np.array([ 90,  90, 100]), np.array([120, 225, 255])),
    "yellow":(np.array([ 10,  70, 140]), np.array([ 40, 135, 255])),
}

# --- カメラセットアップ ---
cap = cv2.VideoCapture(DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print(f"カメラ {DEVICE} を開けませんでした")
    exit()

lower, upper = color_ranges[CALIB_COLOR]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

pixel_diams = []

print(f"キャリブレーション開始: {KNOWN_DISTANCE}cm の位置に {CALIB_COLOR} ボールを置いてください。")
print("エンターキーを押すとサンプリングを開始します。")
input()

print(f"{SAMPLES} フレーム分の測定を行います…")

count = 0
while count < SAMPLES:
    ret, frame = cap.read()
    if not ret:
        continue

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.imshow("Calibration Mask", mask)
        if cv2.waitKey(1) & 0xFF == 27: break
        continue

    # 最大の円を検出
    max_cnt = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(max_cnt)
    if radius < 10:
        # 小さすぎる輪郭は無視
        continue

    pixel_diameter = radius * 2
    pixel_diams.append(pixel_diameter)
    count += 1

    # プレビュー表示
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    cv2.putText(frame, f"{count}/{SAMPLES}: {pixel_diameter:.1f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imshow("Calibration", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

if len(pixel_diams) == 0:
    print("サンプリングに失敗しました。もう一度お試しください。")
    exit()

avg_pixel_diam = sum(pixel_diams) / len(pixel_diams)
# 焦点距離の計算： F = (pixel_diameter * known_distance) / object_diameter
focal_length = (avg_pixel_diam * KNOWN_DISTANCE) / BALL_DIAMETER

print(f"----- キャリブレーション結果 -----")
print(f"サンプル数       : {len(pixel_diams)}")
print(f"平均画素直径     : {avg_pixel_diam:.2f} px")
print(f"計算された焦点距離: {focal_length:.2f} px")
print("プログラム内の FOCAL_LENGTH 定数をこの値に置き換えてください。")

