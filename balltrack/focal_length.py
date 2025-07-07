import cv2
import numpy as np

# 🎯 ボールの実際の直径 [m]
real_ball_diameter = 0.06

# 🎯 実測した距離 [m]（定規などで測って入力）
measured_distance = float(input("実測したカメラとボールの距離 [m] を入力してください："))

# 使用カメラのデバイス指定
DEVICE = '/dev/video4'
cap = cv2.VideoCapture(DEVICE)

if not cap.isOpened():
    print(f"カメラ {DEVICE} を開けませんでした。")
    exit()

# 🎨 赤色のHSV範囲（指定された値）
lower_red = np.array([100, 105, 120])
upper_red = np.array([120, 225, 255])
# ノイズ除去用カーネル
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("赤いボールを中央に映してください。ESCキーで終了します。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 前処理
    blurred = cv2.medianBlur(frame, 11)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 赤色のマスク生成＋ノイズ除去
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_cleaned = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)

    # エッジ検出して輪郭を抽出
    edges = cv2.Canny(mask_cleaned, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_radius = 0
    max_center = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > max_radius:
            max_radius = radius
            max_center = (int(x), int(y))

    if max_center is not None and max_radius > 0:
        image_diameter = 2 * max_radius
        focal_length = (image_diameter * measured_distance) / real_ball_diameter

        # 結果を描画
        cv2.circle(frame, max_center, int(max_radius), (0, 0, 255), 2)
        cv2.circle(frame, max_center, 5, (0, 0, 0), -1)

        cv2.putText(frame, f"Image Diameter: {image_diameter:.2f}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Focal Length: {focal_length:.2f}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        print(f"画像上の直径: {image_diameter:.2f} px")
        print(f"推定焦点距離: {focal_length:.2f} px")

    cv2.imshow("Red Ball Detection (Calibration)", frame)
    if cv2.waitKey(1) == 27:  # ESCキー
        break

cap.release()
cv2.destroyAllWindows()

