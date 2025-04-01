
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Feature extraction parameters
IMAGE_SIZE = (32, 32)
DATASET_PATH = "flowers"
CLASSES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load images and extract features
def load_image_data():
    data = []
    labels = []
    for class_name in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_name)
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img_path = os.path.join(class_path, img_file)
                    img = Image.open(img_path).convert('L')
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img).flatten()
                    data.append(img_array)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
    df = pd.DataFrame(data)
    df['label'] = labels
    return df

# [The rest of the code remains unchanged...]

# We'll append the rest of the code here:



# Metrics
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, fp, tn, fn

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# Naive Bayes Classifier
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0) + 1e-9
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def _pdf(self, x, mean, var):
        numerator = np.exp(-(x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _predict_instance(self, x):
        posteriors = []
        for c in self.classes:
            prior = np.log(self.priors[c])
            class_conditional = np.sum(np.log(self._pdf(x, self.means[c], self.vars[c])))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        return np.array([self._predict_instance(x) for x in X])

# K-Nearest Neighbors
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)

# Logistic Regression
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, num_epochs=1000):
        self.lr = lr
        self.num_epochs = num_epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.num_epochs):
            linear = np.dot(X, self.w) + self.b
            y_pred = self._sigmoid(linear)
            dw = np.dot(X.T, (y_pred - y)) / X.shape[0]
            db = np.mean(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.w) + self.b
        y_pred = self._sigmoid(linear)
        return (y_pred >= 0.5).astype(int)

# Linear SVM
class LinearSVM:
    def __init__(self, lr=0.001, num_epochs=1000, C=1.0):
        self.lr = lr
        self.num_epochs = num_epochs
        self.C = C

    def fit(self, X, y):
        y = np.where(y == 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.num_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) + self.b)
                if condition < 1:
                    self.w -= self.lr * (self.w - self.C * y[idx] * x_i)
                    self.b -= self.lr * (-self.C * y[idx])
                else:
                    self.w -= self.lr * self.w

    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.where(linear_output >= 0, 1, 0)

# Main script
def main():
    df = load_image_data()
    X = df.drop("label", axis=1).values
    y_raw = df["label"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    # To simplify: use binary classification (e.g., only first two classes)
    binary_mask = (y_encoded == 0) | (y_encoded == 1)
    X = X[binary_mask]
    y = y_encoded[binary_mask]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Naive Bayes": NaiveBayesClassifier(),
        "KNN": KNNClassifier(k=3),
        "Logistic Regression": LogisticRegressionScratch(lr=0.01, num_epochs=1000),
        "Linear SVM": LinearSVM(lr=0.001, num_epochs=10, C=1.0)
    }

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"Accuracy:  {accuracy(y_test, y_pred):.3f}")
        print(f"Precision: {precision(y_test, y_pred):.3f}")
        print(f"Recall:    {recall(y_test, y_pred):.3f}")

if __name__ == "__main__":
    main()
