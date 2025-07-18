import cv2

DEVICE_INDEX = 1 # /dev/video4 に対応

cap = cv2.VideoCapture(DEVICE_INDEX)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームの取得に失敗しました")
        break

    cv2.imshow('4K USB Camera', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break

cap.release()
cv2.destroyAllWindows()

