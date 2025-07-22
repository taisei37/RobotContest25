import cv2
import numpy as np
import time

# カメラ設定
cap = cv2.VideoCapture('/dev/video4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

# 想定ボールの中心と半径
true_center = (640, 360)
true_radius = 100  # 画面上での想定ボール半径 [px]
tolerance = 50

def detect_ball_hsv(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    return detect_circle(mask)

def detect_ball_rgb(frame):
    lower_red = np.array([0, 0, 150])
    upper_red = np.array([100, 100, 255])
    mask = cv2.inRange(frame, lower_red, upper_red)
    return detect_circle(mask)

def detect_circle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        if radius > 10:
            return (int(x), int(y)), int(radius)
    return None, None

def is_detected_circle_valid(center, radius, true_center, true_radius, tolerance_px=50, radius_ratio_range=(0.8, 1.2)):
    if center is None or radius is None:
        return False
    dx = center[0] - true_center[0]
    dy = center[1] - true_center[1]
    dist = np.sqrt(dx**2 + dy**2)
    center_valid = dist < tolerance_px
    radius_valid = radius_ratio_range[0] * true_radius <= radius <= radius_ratio_range[1] * true_radius
    return center_valid and radius_valid

# カウント変数
hsv_correct = 0
hsv_total = 0
rgb_correct = 0
rgb_total = 0

print("スペースキーで10秒間計測を開始します")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()
    cv2.circle(disp, true_center, true_radius, (200, 200, 200), 2)
    cv2.putText(disp, "Press SPACE to start 10s test", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Detection", disp)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        start = time.time()
        while time.time() - start < 10:
            ret, frame = cap.read()
            if not ret:
                break

            hsv_center, hsv_radius = detect_ball_hsv(frame)
            rgb_center, rgb_radius = detect_ball_rgb(frame)

            hsv_total += 1
            rgb_total += 1

            if is_detected_circle_valid(hsv_center, hsv_radius, true_center, true_radius):
                hsv_correct += 1
                cv2.circle(frame, hsv_center, hsv_radius, (0, 255, 0), 2)
            if is_detected_circle_valid(rgb_center, rgb_radius, true_center, true_radius):
                rgb_correct += 1
                cv2.circle(frame, rgb_center, rgb_radius, (255, 0, 0), 2)

            # 描画用
            cv2.circle(frame, true_center, true_radius, (200, 200, 200), 2)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) == 27:
                break

        # 結果
        print(f"\n=== 結果 ===")
        print(f"HSV 認識率: {hsv_correct}/{hsv_total} ({(hsv_correct/hsv_total)*100:.2f}%)")
        print(f"RGB 認識率: {rgb_correct}/{rgb_total} ({(rgb_correct/rgb_total)*100:.2f}%)")
        break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()

