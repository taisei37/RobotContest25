import cv2
import numpy as np
import time

# トラックバーのコールバック（何もしない）
def nothing(x):
    pass

# ウィンドウとトラックバー作成
cv2.namedWindow("Controls")
cv2.createTrackbar("H Min", "Controls", 0, 179, nothing)
cv2.createTrackbar("H Max", "Controls", 179, 179, nothing)
cv2.createTrackbar("S Min", "Controls", 0, 255, nothing)
cv2.createTrackbar("S Max", "Controls", 255, 255, nothing)
cv2.createTrackbar("V Min", "Controls", 0, 255, nothing)
cv2.createTrackbar("V Max", "Controls", 255, 255, nothing)

# カメラ起動
cap = cv2.VideoCapture(4)
if not cap.isOpened():
    print("カメラを開けません")
    exit()

box_size = 100
last_update_time = time.time()
hsv_avg = [0, 0, 0]

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できません")
        break

    height, width, _ = frame.shape
    center_x = width // 2
    center_y = height // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 平均HSV計算（2秒に1回）
    current_time = time.time()
    if current_time - last_update_time >= 5.0:
        roi = hsv_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        if roi.size > 0:
            hsv_avg = np.mean(roi.reshape(-1, 3), axis=0).astype(int)
        print(f"[{time.strftime('%H:%M:%S')}] 中心のHSV値: H={hsv_avg[0]}, S={hsv_avg[1]}, V={hsv_avg[2]}")
        last_update_time = current_time

    # トラックバーから値取得
    h_min = cv2.getTrackbarPos("H Min", "Controls")
    h_max = cv2.getTrackbarPos("H Max", "Controls")
    s_min = cv2.getTrackbarPos("S Min", "Controls")
    s_max = cv2.getTrackbarPos("S Max", "Controls")
    v_min = cv2.getTrackbarPos("V Min", "Controls")
    v_max = cv2.getTrackbarPos("V Max", "Controls")

    # HSV範囲のマスクを作成
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_frame, lower, upper)

    # 中心に四角を描く
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # 画面にHSV値表示
    text = f"H: {hsv_avg[0]}  S: {hsv_avg[1]}  V: {hsv_avg[2]}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # 結果表示
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
