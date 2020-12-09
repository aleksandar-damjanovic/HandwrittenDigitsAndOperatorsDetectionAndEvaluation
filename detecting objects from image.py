import cv2
import numpy as np
import os.path
import tensorflow as tf
import operator
import matplotlib.pyplot as plt

mypath = "Test data"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

#image path- enter the image path
img_path = ''

img = cv2.imread(img_path)
print(img.shape)
resized_image = cv2.resize(img, (600, 300))
print(img.shape)
kernel = np.ones((5,5), np.uint8)

#converting images
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (7,7), 0)
canny_image = cv2.Canny(blur_image, 30, 60)
dilation_image = cv2.dilate(canny_image, kernel, iterations=3)
erosion_image = cv2.erode(dilation_image, kernel, iterations=1)
final_image = erosion_image
erosion_image = cv2.rotate(erosion_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
final_resized = cv2.resize(final_image, (25, 25))
original_image = cv2.rotate(resized_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
original_centeroids = cv2.rotate(resized_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
#display image after removed noise


contours, hierarchy = cv2.findContours(erosion_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
i = 1
for cnt in contours:
    train_data_images = []
    train_data = []
    area = cv2.contourArea(cnt)
    if area>100 and area<11000:
        cv2.drawContours(erosion_image, cnt, -1, (0, 255, 0), 3)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.rectangle(original_image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 1)

        im_crop = erosion_image[y-10:y + h + 10, x-10:x + w + 10]
        im_resize = cv2.resize(im_crop, (280, 280))
        im_resize_fit = cv2.resize(im_crop, (28, 28))
        cropped_rotated = cv2.rotate(im_resize, cv2.cv2.ROTATE_90_CLOCKWISE)
        train_data_images.append(cropped_rotated)
        print('Object cropped successfully, image saved.')

        cv2.imwrite('Test data/Testimage' + str(i) + '.jpg', cropped_rotated)
        i += 1
        cv2.imshow('Train data', cropped_rotated)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

print('Number of objects detected: ', len(contours))

#function for identifying centeroid (places a red circle on the centers of detected objects)
def identify_centeroid(original_centeroids, centroid):
    cent_moment =cv2.moments(centroid)
    centroid_x = int(cent_moment['m10'] / cent_moment['m00'])
    centroid_y = int(cent_moment['m01'] / cent_moment['m00'])
    cv2.circle(original_centeroids, (centroid_x, centroid_y), 3, (0, 0, 225), 5)
    return original_centeroids

for (i,c) in enumerate(contours):
    orig = identify_centeroid(original_centeroids, c)

#stacking images
img_hor1 = np.hstack((gray_image, blur_image, canny_image))
img_hor2 = np.hstack((dilation_image, final_image))
img_hor3 = np.hstack((original_image, original_centeroids))
img_hor3 = cv2.rotate(img_hor3, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('First stacked row', img_hor1)
cv2.imshow('Second stacked row', img_hor2)
cv2.imshow('Stacked original and centeroids', img_hor3)
cv2.waitKey(0)
cv2.destroyAllWindows()

