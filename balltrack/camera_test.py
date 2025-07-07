import cv2
import subprocess

# MJPEGフォーマットに変更
subprocess.run([
    'v4l2-ctl', '-d', '/dev/video5',
    '--set-fmt-video=width=1920,height=1080,pixelformat=MJPG'
])

cap = cv2.VideoCapture('/dev/video5')
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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

