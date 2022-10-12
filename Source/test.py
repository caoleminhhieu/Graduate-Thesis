import cv2

img = cv2.imread('data/test_images/test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = cv2.imread('mask/test.jpg')
mask = cv2.cvtColor(img, cv2.COLOR_BGR2RGB, 0)
mask = mask
print(img)
