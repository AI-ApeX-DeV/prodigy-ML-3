import numpy as np
import cv2
from sklearn.svm import SVC

# Load the training data
training_path = "./training/"
data = []
labels = []
for filename in os.listdir(training_path):
    if "dog" in filename:
        labels.append(1)
    else:
        labels.append(0)
    img = cv2.imread(os.path.join(training_path, filename))
    img = cv2.resize(img, (128, 64))
    data.append(img.flatten())

# Create the SVM model
model = SVC()

# Train the model
model.fit(data, labels)

# Load the test data
test_path = "./test/"
data_test = []
labels_test = []
for filename in os.listdir(test_path):
    if "dog" in filename:
        labels_test.append(1)
    else:
        labels_test.append(0)
    img = cv2.imread(os.path.join(test_path, filename))
    img = cv2.resize(img, (128, 64))
    data_test.append(img.flatten())

# Predict the labels for the test data
predictions = model.predict(data_test)

# Calculate the accuracy
accuracy = np.mean(predictions == labels_test)

print("Accuracy:", accuracy)
