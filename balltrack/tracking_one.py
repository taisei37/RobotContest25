import cv2
import numpy as np

cap = cv2.VideoCapture(1)

# HSV色範囲（赤・青・黄）
lower_red = np.array([165, 105, 115])
upper_red = np.array([175, 250, 255])
lower_blue = np.array([90, 90, 100])
upper_blue = np.array([120, 225, 255])
lower_yellow = np.array([10, 70, 140])
upper_yellow = np.array([40, 135, 255])

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

    # 個別のマスク生成
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 結合マスク生成（ビット単位のOR）
    combined_mask = mask_red | mask_blue | mask_yellow

    # 表示用：マスクを別ウィンドウに表示
    cv2.imshow("Combined Mask", combined_mask)

    # マスク辞書（処理用）
    masks = {
        "red": mask_red,
        "blue": mask_blue,
        "yellow": mask_yellow
    }

    max_radius = 0
    max_center = None
    max_color = None

    for color, mask in masks.items():
        # ノイズ除去
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 方法1：HoughCirclesで検出
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask_cleaned)
        circles = cv2.HoughCircles(masked_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                   param1=100, param2=20, minRadius=5, maxRadius=120)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if r > max_radius:
                    max_radius = r
                    max_center = (x, y)
                    max_color = color

        # 方法2：Canny + 輪郭
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

    # 最大円を描画
    if max_center is not None:
        cv2.circle(frame, max_center, int(max_radius), draw_colors[max_color], 2)
        cv2.circle(frame, max_center, 5, (0, 0, 0), -1)
        cv2.putText(frame, f"{max_color.capitalize()} Ball Pos: {max_center}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_colors[max_color], 2)

    cv2.imshow("Hybrid Detection", frame)

    if cv2.waitKey(1) == 27:  # ESCキー
        break

cap.release()
cv2.destroyAllWindows()
