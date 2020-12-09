import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["-", "%", "[", "]", "_", "+", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

file = 'C:/Users/coa/PycharmProjects/TFproject1/Test data/Testimage5.jpg'

def prepare(filepath):
    IMG_SIZE = 28  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

model = tf.keras.models.load_model("C:/Users/coa/AppData/Roaming/JetBrains/PyCharmCE2020.2/scratches/CNN.model100")
prediction = model.predict([prepare(file)])
print(prediction)

y = np.hstack(prediction)
print(y)
g = ([i for i,x in enumerate(y) if x == 1])
print(g)


if g == [0]:
    number = '-'
elif g == [0]:
    number = '/'
elif g == [2]:
    number = '('
elif g == [3]:
    number = ')'
elif g == [4]:
    number = '*'
elif g == [5]:
    number = '+'
elif g == [6]:
    number = '0'
elif g == [7]:
    number = '1'
elif g == [8]:
    number = '2'
elif g == [9]:
    number = '3'
elif g == [10]:
    number = '4'
elif g == [11]:
    number = '5'
elif g == [12]:
    number = '6'
elif g == [13]:
    number = '7'
elif g == [14]:
    number = '8'
elif g == [15]:
    number = '9'
