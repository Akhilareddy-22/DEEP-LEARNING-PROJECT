# =========================
# 1. IMPORT LIBRARIES
# =========================
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2

# =========================
# 2. LOAD DATASET
# =========================
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# =========================
# 3. NORMALIZE DATA
# =========================
X_train = X_train / 255.0
X_test = X_test / 255.0

# =========================
# 4. CLASS NAMES
# =========================
class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

# =========================
# 5. DISPLAY TRAINING IMAGES (CLEAR)
# =========================
plt.figure(figsize=(12,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    img = cv2.resize(X_train[i], (128,128), interpolation=cv2.INTER_CUBIC)
    plt.imshow(img)
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# =========================
# 6. BUILD CNN MODEL
# =========================
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# =========================
# 7. COMPILE MODEL
# =========================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# =========================
# 8. TRAIN MODEL
# =========================
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# =========================
# 9. EVALUATE MODEL
# =========================
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# =========================
# 10. PLOT ACCURACY GRAPH
# =========================
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_graph.png")
plt.show()

# =========================
# 11. PLOT LOSS GRAPH
# =========================
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_graph.png")
plt.show()

# =========================
# 12. MAKE PREDICTIONS
# =========================
predictions = model.predict(X_test)

# =========================
# 13. SHOW CLEAR PREDICTION IMAGE
# =========================
index = 5

img = X_test[index]
img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Predicted: " + class_names[np.argmax(predictions[index])])
plt.axis("off")
plt.savefig("prediction_output.png")
plt.show()

# =========================
# 14. SAVE MODEL
# =========================
model.save("task2_cnn_model.h5")
print("Model saved successfully")