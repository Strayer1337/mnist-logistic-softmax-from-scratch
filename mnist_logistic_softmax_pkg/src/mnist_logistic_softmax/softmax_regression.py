import numpy as np
import matplotlib.pyplot as plt


class SoftmaxRegression:
    def __init__(self, learning_rate=0.5, epochs=300):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    @staticmethod
    def softmax(logits):
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=1))

    @staticmethod
    def one_hot_encode(labels, num_classes):
        one_hot_labels = np.zeros((len(labels), num_classes))
        one_hot_labels[np.arange(len(labels)), labels] = 1
        return one_hot_labels

    def fit(self, features, labels):
        n_samples, n_features = features.shape
        num_classes = len(np.unique(labels))

        y_one_hot = self.one_hot_encode(labels, num_classes)

        self.weights = np.zeros((n_features, num_classes))
        self.bias = np.zeros(num_classes)

        for epoch in range(self.epochs):
            logits = np.dot(features, self.weights) + self.bias
            probabilities = self.softmax(logits)

            loss = self.compute_loss(y_one_hot, probabilities)
            self.losses.append(loss)

            dz = probabilities - y_one_hot
            dw = (1 / n_samples) * np.dot(features.T, dz)
            db = (1 / n_samples) * np.sum(dz, axis=0)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss:.4f}")

    def predict(self, features):
        logits = np.dot(features, self.weights) + self.bias
        probabilities = self.softmax(logits)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, y_true, y_pred):
        num_classes = len(np.unique(y_true))

        precisions, recalls, f1_scores = [], [], []

        for class_index in range(num_classes):
            true_positive = np.sum((y_true == class_index) & (y_pred == class_index))
            false_positive = np.sum((y_true != class_index) & (y_pred == class_index))
            false_negative = np.sum((y_true == class_index) & (y_pred != class_index))

            precision = true_positive / (true_positive + false_positive + 1e-9)
            recall = true_positive / (true_positive + false_negative + 1e-9)
            f1_score = 2 * precision * recall / (precision + recall + 1e-9)

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

        return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Softmax Training Loss")
        plt.show()
