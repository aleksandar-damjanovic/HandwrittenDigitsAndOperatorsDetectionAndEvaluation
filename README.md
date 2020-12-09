Photomath test project

This is a test project for Photomath.
There are 5 files (main, detecting objects from an image, preparing the image, training network 2, and testing model 2.
In main.py you can add a picture to the path, and the program will detect numbers on that picture and recognize the digits. Also, there is a code for parsing and solving expressions.
However, the mnist database does not have models for operators +,-,*,/ and brackets, so there is a test expression you can use. You need to test the model just once, then load saved.
In the "preparing image" file, you can prepare your own file from number and operator images. Files with number images and operators "0,1,2,3,4,5,6,7,8,9,0,/,*,-,+,(,)" are given. 
After preparing images, you can train the model in "training network 2". This model will be saved as CNN.model100. After training the model with over 333000 pictures, 
with 3 epochs and batch_size=64, the total accuracy for every next image was around 92%. In the "testing model 2" file, you can train that model with images that are not yet tested. 
For this project, I decided to keep the mnist database model. 
