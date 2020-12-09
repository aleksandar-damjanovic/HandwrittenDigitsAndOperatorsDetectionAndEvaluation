


image_prepared_for_testing = 'C:/Users/coa/Desktop/Testing images/Test data/Testimage1.jpg'

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_test = tf.keras.utils.normalize(x_test, axis=1)

for test in range(len(image_prepared_for_testing)):
    for row in range(28):
        for x in range(28):
            if x_test[test][row][x] != 0:
                x_test[test][row][x] = 1

model = tf.keras.models.load_model('CNN.model')
print(len(image_prepared_for_testing))
predictions = model.predict(image_prepared_for_testing)

count = 0
for x in range(len(image_prepared_for_testing)):
    guess = (np.argmax(predictions[x]))
    print("I predict this number is a:", guess)

    plt.imshow(image_prepared_for_testing, cmap=plt.cm.binary)
    plt.show()

print("The program got", count, 'wrong, out of', len(image_prepared_for_testing))
print(str(100 - ((count/len(image_prepared_for_testing))*100)) + '% correct')
print(len(image_prepared_for_testing))
