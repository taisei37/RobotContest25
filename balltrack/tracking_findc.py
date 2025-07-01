import cv2
import numpy as np

cap = cv2.VideoCapture(0)

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
# カーネル定義（モルフォロジー処理用）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 前処理
    blurred = cv2.medianBlur(frame, 11)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    masks = {
        "red": cv2.inRange(hsv, lower_red, upper_red),
        "blue": cv2.inRange(hsv, lower_blue, upper_blue),
        "yellow": cv2.inRange(hsv, lower_yellow, upper_yellow)
    }

    # 最大円を記録する変数
    max_radius = 0
    max_center = None
    max_color = None

    for color, mask in masks.items():
        # ノイズ除去
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # Cannyエッジ検出
        edges = cv2.Canny(mask, 50, 150)

        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 小さい輪郭の削除
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 300:
                continue

            # 最小外接円を取得
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius > max_radius:
                max_radius = radius
                max_center = (int(x), int(y))
                max_color = color

    # 最大の円を描画
    if max_center is not None:
        cv2.circle(frame, max_center, int(max_radius), draw_colors[max_color], 2)
        cv2.circle(frame, max_center, 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{max_color.capitalize()} Ball Pos: {max_center}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_color], 2)

    # 表示
    cv2.imshow("Canny + Contour", frame)

    if cv2.waitKey(1) == 27:  # ESCキー
        break

cap.release()
cv2.destroyAllWindows()
