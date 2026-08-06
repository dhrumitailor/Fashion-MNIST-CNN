# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image

# # Load trained model
# model = tf.keras.models.load_model("cnn_model.keras")

# # Class names
# class_names = [
#     "T-shirt/Top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle Boot"
# ]

# st.title("👕 Fashion MNIST Classifier")

# st.write("Upload a grayscale clothing image (28×28 pixels).")

# uploaded_file = st.file_uploader(
#     "Choose an image",
#     type=["png", "jpg", "jpeg"]
# )

# if uploaded_file is not None:

#     image = Image.open(uploaded_file)

#     st.image(image, caption="Uploaded Image", width=200)

#     image = image.convert("L")
#     image = image.resize((28, 28))

#     image_array = np.array(image)

#     image_array = image_array / 255.0

#     image_array = image_array.reshape(1, 28, 28, 1)

#     prediction = model.predict(image_array)

#     predicted_class = np.argmax(prediction)

#     confidence = np.max(prediction) * 100

#     st.success(
#         f"Prediction : {class_names[predicted_class]}"
#     )

#     st.info(
#         f"Confidence : {confidence:.2f}%"
#     )

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load Model
# -------------------------------

model = tf.keras.models.load_model("cnn_model.keras")

# -------------------------------
# Load Fashion-MNIST Test Dataset
# -------------------------------

fashion_mnist = tf.keras.datasets.fashion_mnist

(_, _), (x_test, y_test) = fashion_mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# -------------------------------
# Class Names
# -------------------------------

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

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

st.title("👕 Fashion-MNIST Clothes Classifier")

st.write(
    "This CNN was trained on the Fashion-MNIST dataset."
)

st.write(
    "Select any test image and let the model classify it."
)

# -------------------------------
# Select Image
# -------------------------------

image_index = st.slider(
    "Choose Test Image",
    0,
    len(x_test)-1,
    0
)

image = x_test[image_index]

actual_label = class_names[y_test[image_index]]

# -------------------------------
# Display Image
# -------------------------------

fig, ax = plt.subplots()

ax.imshow(image.reshape(28,28), cmap="gray")
ax.axis("off")

st.pyplot(fig)

# -------------------------------
# Prediction
# -------------------------------

prediction = model.predict(
    image.reshape(1,28,28,1),
    verbose=0
)

predicted_class = np.argmax(prediction)

confidence = prediction[0][predicted_class] * 100

st.success(
    f"Predicted : {class_names[predicted_class]}"
)

st.info(
    f"Confidence : {confidence:.2f}%"
)

st.write(
    f"Actual Label : **{actual_label}**"
)

# -------------------------------
# Top 5 Predictions
# -------------------------------

st.subheader("Top 5 Predictions")

top5 = np.argsort(prediction[0])[::-1][:5]

for idx in top5:

    st.progress(float(prediction[0][idx]))

    st.write(
        f"{class_names[idx]} : {prediction[0][idx]*100:.2f}%"
    )