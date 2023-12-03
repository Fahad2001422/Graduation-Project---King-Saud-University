# Import necessary libraries
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
normal_data_dir = "C:\\Users\\Alhak\\Downloads\\CSC496\\DeepLungCodes\\NormalPneumonia"
ill_data_dir = "C:\\Users\\Alhak\\Downloads\\CSC496\\DeepLungCodes\\IllPneumonia"
image_size = (256, 256)
# Load the images and labels 
data = []
labels = [] 
class_names = ["normal", "ill"]

# Load normal images
normal_files = os.listdir(normal_data_dir)
for img_file in normal_files:
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(normal_data_dir, img_file), target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    data.append(img_array)
    labels.append(class_names.index("normal"))

# Load ill images
ill_files = os.listdir(ill_data_dir)
for img_file in ill_files:
    img = tf.keras.preprocessing.image.load_img(
        os.path.join(ill_data_dir, img_file), target_size=image_size
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    data.append(img_array)
    labels.append(class_names.index("ill"))

# Convert the data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# Split your dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# Load VGG16 without the top layer (which consists of fully connected layers)
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(image_size[0], image_size[1], 3))

# Make the layers of VGG16 non-trainable
for layer in vgg16.layers:
    layer.trainable = False

# Create a new model
model = Sequential()

# Add the VGG16 base model
model.add(vgg16)

# Add new layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # Binary classification

# Compile the model
precision = Precision(name='precision')
recall = Recall(name='recall')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[precision, recall])

# Train the model
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=10,
                    validation_data=(X_test, y_test))

# Calculate F1 score
precision = np.mean(history.history['val_precision'])
recall = np.mean(history.history['val_recall'])
f1_scores = [2 * (precision * recall) / (precision + recall) 
             for precision, recall in zip(history.history['val_precision'], history.history['val_recall'])]
print('\nMax Validation F1 Score:', max(f1_scores)) 
model.save("C:\\Users\\Alhak\\Downloads\\CSC496\\my_PneumoniaModel.h5")