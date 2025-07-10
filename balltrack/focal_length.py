import cv2
import numpy as np

# 🎯 ボールの実際の直径（単位：メートル）
REAL_BALL_DIAMETER = 0.065 

# 📷 カメラデバイス
DEVICE = '/dev/video4'

# 🔍 赤色のHSV範囲（より一般的な範囲に修正）
LOWER_RED = np.array([165, 105, 115])
UPPER_RED = np.array([175, 255, 255])

# 🔧 ノイズ除去カーネル
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def calculate_focal_length(image_diameter_px, real_diameter_m, distance_m):
    return (image_diameter_px * distance_m) / real_diameter_m

def detect_largest_red_circle(frame):
    blurred = cv2.medianBlur(frame, 11)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)

    edges = cv2.Canny(cleaned, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_radius = 0
    max_center = None

    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > max_radius:
            max_radius = radius
            max_center = (int(x), int(y))

    if max_center:
        return 2 * max_radius, max_center, int(max_radius)
    else:
        return None, None, None

def main():
    measured_distance = float(input("📏 実測したカメラとボールの距離 [m]："))

    cap = cv2.VideoCapture(DEVICE)
    if not cap.isOpened():
        print(f"❌ カメラ {DEVICE} を開けませんでした。")
        return

    print("📸 赤いボールを中央に映してください。ESCキーで終了します。")

    focal_lengths = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_diameter, center, radius = detect_largest_red_circle(frame)

        if image_diameter:
            focal = calculate_focal_length(image_diameter, REAL_BALL_DIAMETER, measured_distance)
            focal_lengths.append(focal)

            cv2.circle(frame, center, radius, (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 0), -1)

            cv2.putText(frame, f"Diameter: {image_diameter:.2f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Focal Length: {focal:.2f}px", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(" Red Ball Detection (Calibration)", frame)

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

    if focal_lengths:
        avg_focal = np.mean(focal_lengths)
        print(f"\n✅ 平均焦点距離: {avg_focal:.2f} px（{len(focal_lengths)}回の測定）")
    else:
        print("⚠️ 有効なボールが検出されませんでした。")

if __name__ == '__main__':
    main()

