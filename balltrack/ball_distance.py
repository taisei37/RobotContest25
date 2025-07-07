import cv2
import numpy as np

# 実際のボールの直径 [m]（例: 6cm）
real_ball_diameter = 0.065 

# カメラの焦点距離 [pixel]（キャリブレーションして算出する。）
focal_length =  455 

DEVICE = '/dev/video4'  
cap = cv2.VideoCapture(DEVICE)

# HSV色範囲（赤・青・黄）
lower_red = np.array([145, 120, 120])
upper_red = np.array([165, 240, 255])
lower_blue = np.array([100, 105, 120])
upper_blue = np.array([120, 225, 255])
lower_yellow = np.array([0, 28, 108])
upper_yellow = np.array([35, 255, 255])

draw_colors = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.medianBlur(frame, 11)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    masks = {
        "red": mask_red,
        "blue": mask_blue,
        "yellow": mask_yellow
    }

    max_radius = 0
    max_center = None
    max_color = None

    for color, mask in masks.items():
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        edges = cv2.Canny(mask_cleaned, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > max_radius:
                max_radius = radius
                max_center = (int(x), int(y))
                max_color = color

    if max_center is not None:
        # 半径があれば距離計算（2 * radius = 画像上の直径）
        if max_radius > 0:
            image_diameter = 2 * max_radius
            distance = (real_ball_diameter * focal_length) / image_diameter

            cv2.circle(frame, max_center, int(max_radius), draw_colors[max_color], 2)
            cv2.circle(frame, max_center, 5, (0, 0, 0), -1)

            # テキスト表示
            cv2.putText(frame, f"{max_color.capitalize()} Pos: {max_center}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_color], 2)
            cv2.putText(frame, f"Distance: {distance:.2f} m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_color], 2)

    cv2.imshow("Ball Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

