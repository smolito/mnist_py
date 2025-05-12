import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from conv.conv_model import MNISTClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random

'''
docker build -t mnist-tf-gpu .
docker run --gpus all -it --rm -v "$(pwd):/mnist_py" mnist-tf-gpu
'''

# 1. load the MNIST dataset: data, labels
# https://keras.io/api/datasets/mnist/

# cuda troubleshooting
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"training set: {x_train.shape[0]} images")
print(f"testing set: {x_test.shape[0]} images") 

# 2. plot randomly selected images from the training set

plt.figure(figsize=(10, 10))
for i in range(40):
    rand_idx = np.random.randint(0, len(x_train))
    plt.subplot(5, 8, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[rand_idx], cmap=plt.cm.binary)
    plt.xlabel(y_train[rand_idx])
plt.savefig("output_image.png")

# 3. num of unique labels and their counts
unique, counts = np.unique(y_train, return_counts=True)
print("images per label:")
for u, c in zip(unique, counts):
    print(f"label {u}: {c} images")

# 4. img sizes
print(f"train images size: {x_train[0].shape}")
print(f"test images size: {x_test[0].shape}")

# 5. split the TRAINING set into train and validation sets
# testing set is a standalone set
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    test_size=0.4,
    random_state=1234
)

# data augmentation ???

# model summary and CUDA training
# 6. build the model
model = MNISTClassifier()
model.build_model()

# output shape of the model -> batch size, height, width, channels
print("model summary: layers, output shape [batch size, height, width, channels], num of params")
model.model.summary()

# 7. setup training parameters
history = model.model.fit(
    x_train, y_train,
    batch_size=256,
    epochs=13,
    validation_data=(x_val, y_val),
    verbose=1,
)

# 8. model evaluation
test_loss, test_acc = model.model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# predictions for the test set
y_pred = model.model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# random image from the test set
random_idx = random.randint(0, len(x_test) - 1)
random_image = x_test[random_idx]

random_image_input = np.expand_dims(random_image, axis=0)
softmax_probs = model.model.predict(random_image_input)[0]
predicted_class = np.argmax(softmax_probs)

# Plot the random image
plt.figure()
plt.imshow(random_image, cmap=plt.cm.binary)
plt.title(f"Predicted Class: {predicted_class}")
plt.axis("off")
plt.savefig("random_image.png")
plt.close()

# predicted class and probabilities
print(f"Predicted class for the randomly selected image is: {predicted_class}")
print("Softmax probability distribution:")
for i, prob in enumerate(softmax_probs):
    print(f"Class {i}: {prob:.4f}")
