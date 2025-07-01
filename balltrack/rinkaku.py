import cv2

# 画像読み込みと前処理
img = cv2.imread('apple.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 赤色のマスク（二値化）
lower_red1 = (0, 100, 100)
upper_red1 = (10, 255, 255)
lower_red2 = (160, 100, 100)
upper_red2 = (180, 255, 255)
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# 輪郭検出
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 輪郭を描画
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# 表示
cv2.imshow("Original", img)
cv2.imshow("Mask (Binary)", mask)
cv2.imshow("Contours", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
