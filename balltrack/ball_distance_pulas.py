import cv2
import numpy as np
import time

# カメラ設定
DEVICE = '/dev/video4'
cap = cv2.VideoCapture(DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print(f"カメラ {DEVICE} を開けませんでした")
    exit()

# 実際のボール直径（cm）
BALL_DIAMETER = 5.5  

# カメラキャリブレーションパラメータ
camera_matrix = np.array([
    [750.10059546,   0.0,        704.54913907],
    [0.0,            746.54075486, 445.7714058],
    [0.0,            0.0,          1.0]
])

dist_coeffs = np.array([0.04739503, -0.07422041, 0.00880341, 0.0123376, 0.02295108])

# HSV色範囲（赤・青・黄）
color_ranges = {
    "red": (np.array([165, 105, 115]), np.array([175, 250, 255])),
    "blue": (np.array([90, 90, 100]), np.array([120, 225, 255])),
    "yellow": (np.array([10, 70, 140]), np.array([40, 135, 255]))
}

draw_colors = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def update_max_circle(x, y, radius, color, current_max):
    if radius > current_max["radius"]:
        current_max["radius"] = radius
        current_max["center"] = (int(x), int(y))
        current_max["color"] = color
    return current_max

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("フレーム取得に失敗")
        break

    # 歪み補正
    frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

    blurred = cv2.medianBlur(frame_undistorted, 5)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    combined_mask = None
    masks = {}

    max_circle = {
        "radius": 0,
        "center": None,
        "color": None
    }

    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        masks[color] = mask
        combined_mask = mask if combined_mask is None else combined_mask | mask

    for color, mask in masks.items():
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            max_circle = update_max_circle(x, y, radius, color, max_circle)

    if max_circle["center"] is not None:
        cv2.circle(frame_undistorted, max_circle["center"], int(max_circle["radius"]), draw_colors[max_circle["color"]], 2)
        cv2.circle(frame_undistorted, max_circle["center"], 5, (0, 0, 0), -1)
        cv2.putText(frame_undistorted, f"{max_circle['color'].capitalize()} Ball Pos: {max_circle['center']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_circle["color"]], 2)

        # 距離計算（焦点距離は fx を使用）
        pixel_diameter = max_circle["radius"] * 2
        if pixel_diameter > 0:
            fx = camera_matrix[0, 0]
            distance = (BALL_DIAMETER * fx) / pixel_diameter
            cv2.putText(frame_undistorted, f"Distance: {distance:.2f} cm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_circle["color"]], 2)

    # 表示
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Undistorted View", frame_undistorted)

    fps = 1 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}", end='\r')

    time.sleep(0.1)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


