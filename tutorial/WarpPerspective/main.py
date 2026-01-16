import cv2
import numpy as np

img = cv2.imread("cv/tutorial/WarpPerspective/image.png")

width, height = 250, 350

points1 = np.array([
    [518, 435],
    [1368, 359],
    [694, 1708],
    [1613, 1549]
], dtype=np.float32)

points2 = np.array([
    [0, 0],            # top-left
    [width, 0],        # top-right
    [0, height],       # bottom-left
    [width, height]    # bottom-right
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(points1, points2)
ImgOutput = cv2.warpPerspective(img, matrix, (width, height))


cv2.circle(img, (int(points1[0][0]), int(points1[0][1])), 2, (0,255,0), 2)

cv2.circle(img, (int(points1[1][0]), int(points1[1][1])), 2, (0,255,0), 2)

cv2.circle(img, (int(points1[2][0]), int(points1[2][1])), 2, (0,255,0), 2)

cv2.circle(img, (int(points1[3][0]), int(points1[3][1])), 2, (0,255,0), 2)

cv2.namedWindow("Original Img", cv2.WINDOW_NORMAL)
cv2.imshow("Original Img", img)

cv2.namedWindow("Warp Img", cv2.WINDOW_NORMAL)
cv2.imshow("Warp Img", ImgOutput)

cv2.waitKey(0)
