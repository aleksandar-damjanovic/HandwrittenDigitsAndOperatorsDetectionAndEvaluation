import cv2
import numpy as np
import  os.path
import tensorflow as tf
import operator
import matplotlib.pyplot as plt
import pandas as pd

mypath = "Test data"
for root, dirs, files in os.walk(mypath):
    for file in files:
        os.remove(os.path.join(root, file))

#image path
img_path = (r'C:\Users\coa\Desktop\Testing images\clear images\all.jpg')
img = cv2.imread(img_path)
print(img.shape)
resized_image = cv2.resize(img, (600, 300))
print(img.shape)
kernel = np.ones((5,5), np.uint8)

#converting images (gray, blur, canny, dilation, erosion
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (7,7), 0)
canny_image = cv2.Canny(blur_image, 30, 60)
dilation_image = cv2.dilate(canny_image, kernel, iterations=5)
erosion_image = cv2.erode(dilation_image, kernel, iterations=2)
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

        im_crop = erosion_image[y-20:y + h + 20, x-20:x + w + 20]
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

#function for identifying centeroid (places a red circle on the centers of contours)
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




# -------------------------- MODEL FROM MNIST DATABASE ------------------------------


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255


# -------------------------- CREATE MODEL ------------------------------

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.save('MNIST_MODEL.model')
print("Model saved")

# ----------------------------------------------------------------------


path, dirs, files = next(os.walk('Test data'))
file_count = len(files)
print('U mapi je slika:', len(files))

model = tf.keras.models.load_model('MNIST_MODEL.model')

numbers = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '*', '/', '+', '-', '(', ')')
objects_detected = []
a = 1
for x in range(len(files)):
    directory = 'Test data/Testimage' + str(a) + '.jpg'
    image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = np.reshape(image, [1, 28, 28, 1])
    image /= 255
    pred = model.predict(image)
    guess = (np.argmax(pred))
    print('I predict this number is ' + '{}'.format(guess), '.')
    objects_detected.append(str(guess))

    plt.imshow(image.reshape(28, 28), cmap='Greys')
    plt.show()
    a = a + 1

print(objects_detected)
expression = (''.join(objects_detected))
print(expression)

# ----------------------------------------------------------------------
# THERE IS NO +,-,*,/,/ IN MNIST, SO THIS IS A TEST EXPRESSION JUST TO TEST PARSE FUNCTION
# ----------------------------------------------------------------------

expression = '7*7+2+2/3-4*5-6'
def parse(x):
    operators = set('+-*/')
    op_out = []
    num_out = []
    buff = []
    for c in x:
        if c in operators:
            num_out.append(''.join(buff))
            buff = []
            op_out.append(c)
        else:
            buff.append(c)
    num_out.append(''.join(buff))
    return num_out,op_out

print(parse(expression))

def my_eval(nums,ops):

    nums = list(nums)
    ops = list(ops)
    operator_order = ('*/','+-')

    op_dict = {'*':operator.mul,
               '/':operator.truediv,
               '+':operator.add,
               '-':operator.sub}
    Value = None
    for op in operator_order:
        while any(o in ops for o in op):
            idx,oo = next((i,o) for i,o in enumerate(ops) if o in op)
            ops.pop(idx)
            values = map(float,nums[idx:idx+2])
            value = op_dict[oo](*values)
            nums[idx:idx+2] = [value]

    return nums[0]

print ('Result of text expression is:', (my_eval(*parse(expression))))
