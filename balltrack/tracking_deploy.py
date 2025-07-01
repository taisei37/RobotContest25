import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# HSV範囲（赤・青・黄）
color_ranges = {
    "red": [([150, 120, 0], [175, 255, 255])],
    "blue": [([95, 195, 0], [125, 255, 255])],
    "yellow": [([20, 34, 205], [30, 88, 255])]
}

draw_colors = {
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}
# カーネル設定
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# マスク作成
def create_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) #モルフォロジー処理により輪郭検出

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.medianBlur(frame, 11) #メディアンブラーによりノイズの除去
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV) # HSVへ変換変換
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) # グレースケール化

    max_radius = 0
    max_center = None
    max_color = None

    for color, ranges in color_ranges.items():
        mask = create_mask(hsv, ranges)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        display_masked = cv2.cvtColor(masked_gray, cv2.COLOR_GRAY2BGR)
        cv2.imshow(f"Hough Input - {color}", display_masked)

        circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                   param1=100, param2=20, minRadius=5, maxRadius=120)
        if circles is not None:
            for (x, y, r) in np.round(circles[0, :]).astype("int"):
                if r > max_radius:
                    max_radius = r
                    max_center = (x, y)
                    max_color = color

    if max_center is not None:
        cv2.circle(frame, max_center, max_radius, draw_colors[max_color], 2)
        cv2.circle(frame, max_center, 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{max_color.capitalize()} Ball Pos: {max_center}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_color], 2)

    cv2.imshow("Final Result", frame)

    if cv2.waitKey(1) == 27:  # ESCキー
        break

cap.release()
cv2.destroyAllWindows()
