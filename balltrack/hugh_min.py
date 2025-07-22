import cv2
import numpy as np
import time

# カメラ設定
cap = cv2.VideoCapture('/dev/video4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

true_center = (640, 360)
true_radius = 100  # px
tolerance = 50     # 中心の許容誤差(px)

def detect_circle_min_enclosing(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
        if radius > 10:
            return (int(x), int(y)), int(radius)
    return None, None

def detect_circle_hough(mask):
    # ハフ関数はグレースケール画像で実行
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 一番大きい円を返す
        largest = max(circles[0, :], key=lambda c: c[2])
        center = (largest[0], largest[1])
        radius = largest[2]
        return center, radius
    return None, None

def is_circle_valid(center, radius, true_center, true_radius, tol=50, ratio_range=(0.8,1.2)):
    if center is None or radius is None:
        return False
    dist = np.linalg.norm(np.array(center) - np.array(true_center))
    radius_ok = ratio_range[0] * true_radius <= radius <= ratio_range[1] * true_radius
    return dist <= tol and radius_ok

print("スペースキーで10秒間計測開始。ESCで終了。")

measuring = False
start_time = None
total_frames = 0
enclosing_correct = 0
hough_correct = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    disp = frame.copy()
    cv2.circle(disp, true_center, true_radius, (200, 200, 200), 2)

    key = cv2.waitKey(1) & 0xFF
    current_time = time.time()

    if not measuring:
        cv2.putText(disp, "Press SPACE to start 10-second test", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if key == 32:
            measuring = True
            start_time = current_time
            total_frames = 0
            enclosing_correct = 0
            hough_correct = 0
            print("計測開始！")
    else:
        elapsed = current_time - start_time
        cv2.putText(disp, f"Measuring... {elapsed:.1f}/10.0 sec", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # HSVマスク処理（赤色範囲）
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # ２手法で検出
        center_enclosing, radius_enclosing = detect_circle_min_enclosing(mask)
        center_hough, radius_hough = detect_circle_hough(mask)

        total_frames += 1
        if is_circle_valid(center_enclosing, radius_enclosing, true_center, true_radius, tolerance):
            enclosing_correct += 1
            cv2.circle(disp, center_enclosing, radius_enclosing, (0, 255, 0), 3)
            cv2.putText(disp, "MinEnclosing", (center_enclosing[0]-60, center_enclosing[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if is_circle_valid(center_hough, radius_hough, true_center, true_radius, tolerance):
            hough_correct += 1
            cv2.circle(disp, center_hough, radius_hough, (255, 0, 0), 3)
            cv2.putText(disp, "Hough", (center_hough[0]-40, center_hough[1]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        if elapsed >= 10:
            measuring = False
            print("\n=== 計測結果 ===")
            print(f"MinEnclosing認識率: {enclosing_correct} / {total_frames} = {(enclosing_correct / total_frames)*100:.2f}%")
            print(f"Hough認識率: {hough_correct} / {total_frames} = {(hough_correct / total_frames)*100:.2f}%")

    cv2.imshow("Circle Detection Comparison", disp)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

