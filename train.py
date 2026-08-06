
# 1. Import Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

# print("TensorFlow Version:", tf.__version__)




# phase 2 
# Load Fashion-MNIST dataset

# 2. Load Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


print("Training Images Shape :", x_train.shape)
print("Training Labels Shape :", y_train.shape)

print("Testing Images Shape :", x_test.shape)
print("Testing Labels Shape :", y_test.shape)

print(type(x_train))
print(type(y_train))

print(x_train[0])
print(y_train[0])

class_names = [
    "T-shirt/Top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# 3. Explore Dataset

print("First Image Label :", class_names[y_train[0]])

plt.imshow(x_train[0], cmap="gray")
plt.title(class_names[y_train[0]])
plt.axis("off")
plt.show()


plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()


# phase 3- 4.Data Preprocessing

print("Before Normalization:")
print(x_train[0][0][:10])

# Normalize pixel values
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nAfter Normalization:")
print(x_train[0][0][:10])

print(x_train.shape)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)

# phase 4  # 5. BUILD CNN  ← Add the new code here
# # Create CNN model

from tensorflow.keras import Input

model = Sequential()

model.add(Input(shape=(28, 28, 1)))

# -------- First Block --------
model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu"
    )
)

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# -------- Second Block --------
model.add(
    Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    )
)
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Phase 5: Flatten and Dense Layers
# -------- Classifier --------
model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation="softmax"))


model.summary()

# phase 6 -Compile the Model

# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )
# # phase 7- testing
# history = model.fit(
#     x_train,
#     y_train,
#     epochs=15,
#     batch_size=64,
#     validation_split=0.2
# )

# # Phase 8: Evaluate the Model
# # Evaluate the model on test data
# test_loss, test_accuracy = model.evaluate(x_test, y_test)


# #9 step 1
# # Predict classes for test images
# predictions = model.predict(x_test)

# predicted_labels = np.argmax(predictions, axis=1)
# # step 2
# cm = confusion_matrix(y_test, predicted_labels)

# print(cm)


# # step 3
# plt.figure(figsize=(10, 8))

# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=class_names,
#     yticklabels=class_names
# )

# plt.xlabel("Predicted Label")
# plt.ylabel("Actual Label")
# plt.title("Fashion-MNIST Confusion Matrix")

# plt.show()


# # step 4
# print(classification_report(
#     y_test,
#     predicted_labels,
#     target_names=class_names
# ))

# print("\nTest Loss :", test_loss)
# print("Test Accuracy :", test_accuracy)


# # remaining tasks
# # Step 1: Plot Accuracy Graph

# plt.figure(figsize=(8,5))

# plt.plot(history.history["accuracy"], label="Training Accuracy")
# plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

# plt.title("Model Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")

# plt.legend()

# plt.show()

# # Step 2: Plot Loss Graph
# plt.figure(figsize=(8,5))

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")

# plt.title("Model Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")

# plt.legend()

# plt.show()

# # Step 3: Save the Model
# model.save("cnn_model.keras")

# print("Model Saved Successfully!")

# # Step 4: Show Predictions on Random Images
# plt.figure(figsize=(12,8))

# for i in range(9):
#     plt.subplot(3,3,i+1)

#     plt.imshow(x_test[i].reshape(28,28), cmap="gray")

#     actual = class_names[y_test[i]]
#     predicted = class_names[predicted_labels[i]]

#     plt.title(f"A:{actual}\nP:{predicted}")

#     plt.axis("off")

# plt.tight_layout()

# plt.show()

# to improve accuracy to ~92–93%
# # 1. Add EarlyStopping
# from tensorflow.keras.callbacks import EarlyStopping

# early_stop = EarlyStopping(
#     monitor="val_loss",
#     patience=3,
#     restore_best_weights=True
# )
# history = model.fit(
#     x_train,
#     y_train,
#     epochs=20,
#     batch_size=64,
#     validation_split=0.2,
#     callbacks=[early_stop]
# )


# from tensorflow.keras.callbacks import ModelCheckpoint

# checkpoint = ModelCheckpoint(
#     "best_model.keras",
#     save_best_only=True,
#     monitor="val_accuracy"
# )
# callbacks=[early_stop, checkpoint]

# phase 6 -Compile the Model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# to improve accuracy to ~92–93%
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    save_best_only=True,
    monitor="val_accuracy"
)

# phase 7- training

history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint]
)

# Phase 8: Evaluate the Model

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Loss :", test_loss)
print("Test Accuracy :", test_accuracy)


#9 step 1
# Predict classes for test images

predictions = model.predict(x_test)

predicted_labels = np.argmax(predictions, axis=1)


# step 2

cm = confusion_matrix(y_test, predicted_labels)

print(cm)


# step 3

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Fashion-MNIST Confusion Matrix")

plt.show()


# step 4

print(classification_report(
    y_test,
    predicted_labels,
    target_names=class_names
))


# remaining tasks

# Step 1: Plot Accuracy Graph

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()

plt.show()


# Step 2: Plot Loss Graph

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.show()


# Step 3: Save the Model

model.save("cnn_model.keras")

print("Model Saved Successfully!")


# Step 4: Show Predictions on Random Images

plt.figure(figsize=(12,8))

for i in range(9):
    plt.subplot(3,3,i+1)

    plt.imshow(x_test[i].reshape(28,28), cmap="gray")

    actual = class_names[y_test[i]]
    predicted = class_names[predicted_labels[i]]

    plt.title(f"A:{actual}\nP:{predicted}")

    plt.axis("off")

plt.tight_layout()

plt.show()