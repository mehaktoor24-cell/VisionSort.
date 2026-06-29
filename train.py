import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# Image preparation
data = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)


# Training data
train_data = data.flow_from_directory(
    "dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)


# Validation data
test_data = data.flow_from_directory(
    "dataset",
    target_size=(150,150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)


# CNN Model
model = Sequential()


model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(150,150,3)
    )
)

model.add(MaxPooling2D())


model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D())


model.add(Flatten())


model.add(
    Dense(
        128,
        activation="relu"
    )
)


model.add(
    Dense(
        3,
        activation="softmax"
    )
)


# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# Train model
model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)


# Save model
model.save("model/visionsort.h5")


print("✅ VisionSort Model Created Successfully")