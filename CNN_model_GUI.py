# Importing useful packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import pickle

# Creating our object-model
model = Sequential()

# 1) layer
model.add(Conv2D(32, (3, 3), strides=1, padding="same", input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))

# 2) layer
model.add(Conv2D(64, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))

# 3) layer
model.add(Conv2D(128, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))

# 4) layer
model.add(Conv2D(256, (3, 3), strides=1, padding="same"))
model.add(Conv2D(256, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))

# 5) layer
model.add(Conv2D(512, (3, 3), strides=1, padding="same"))
model.add(Conv2D(512, (3, 3), strides=1, padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=3, strides=2, padding="same"))

# 6) layer
model.add(AveragePooling2D(pool_size=1, strides=1, padding="valid"))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(15, activation="softmax"))
print(model.summary())

# Compiling our model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='Adam')

# Loading the data from directory
DATADIR = r"C:\Users\adorr\Desktop\cnn\resize_plant_leaf_dataset15"

CATEGORIES = ["Pepperbell_bacterial_spot", "Pepperbell_healthy", "Potato_early_blight", "Potato_healthy",
              "Potato_late_blight", "Tomato_bacterial_spot", "Tomato_early_blight", "Tomato_healthy",
              "Tomato_late_blight", "Tomato_leaf_mold", "Tomato_septoria_leaf_spot", "Tomato_target_spot",
              "Tomato-mosaic_virus", "Tomato-spider_mites_2_spotted_spider_mite", "Tomato-yellow_leaf_curl_virus"]

dataset = list()
categories = list()
categories_name = list()
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        dataset.append(img_array)
        if category == "Pepperbell_bacterial_spot":
            categories.append(0)
            categories_name.append(category)
        elif category == "Pepperbell_healthy":
            categories.append(1)
            categories_name.append(category)
        elif category == "Potato_early_blight":
            categories.append(2)
            categories_name.append(category)
        elif category == "Potato_healthy":
            categories.append(3)
            categories_name.append(category)
        elif category == "Potato_late_blight":
            categories.append(4)
            categories_name.append(category)
        elif category == "Tomato_bacterial_spot":
            categories.append(5)
            categories_name.append(category)
        elif category == "Tomato_early_blight":
            categories.append(6)
            categories_name.append(category)
        elif category == "Tomato_healthy":
            categories.append(7)
            categories_name.append(category)
        elif category == "Tomato_late_blight":
            categories.append(8)
            categories_name.append(category)
        elif category == "Tomato_leaf_mold":
            categories.append(9)
            categories_name.append(category)
        elif category == "Tomato_septoria_leaf_spot":
            categories.append(10)
            categories_name.append(category)
        elif category == "Tomato_target_spot":
            categories.append(11)
            categories_name.append(category)
        elif category == "Tomato-mosaic_virus":
            categories.append(12)
            categories_name.append(category)
        elif category == "Tomato-spider_mites_2_spotted_spider_mite":
            categories.append(13)
            categories_name.append(category)
        elif category == "Tomato-yellow_leaf_curl_virus":
            categories.append(14)
            categories_name.append(category)

# Transform lists to arrays to be processed
categories = np.asarray(categories)
categories_name = np.asarray(categories_name)
dataset_2 = np.asarray(dataset)

# Splitting the data with train_test_split function
X_train, X_test, y_train, y_test = train_test_split(dataset_2, categories, test_size=0.30, random_state=101)

# resizing our arrays
X_test = X_test / 255.0
X_train = X_train / 255.0

# Fitting the data to our model
history = model.fit(X_train, y_train, epochs=20)

# saving our model to filename
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))

# Evaluating the performance of our model
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=1)

print('\nTest accuracy:', test_acc)

# Creating loss and accuracy curves of data
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(test_loss, label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(test_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()

# Making the predictions and transform
# to actual values-categories with np.argmax
predictions = model.predict(X_test)
y_pred = [np.argmax(probas) for probas in predictions]

print(classification_report(y_test, y_pred))

# Creating a nice visualization of the confusion matrix of our model


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CATEGORIES, title='Normalized confusion matrix')
plt.show()




