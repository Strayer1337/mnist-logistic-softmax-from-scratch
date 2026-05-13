import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        return -np.mean(
            y_true * np.log(y_pred + epsilon)
            + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )

    def fit(self, features, labels):
        n_samples, n_features = features.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_output = np.dot(features, self.weights) + self.bias
            probabilities = self.sigmoid(linear_output)

            loss = self.compute_loss(labels, probabilities)
            self.losses.append(loss)

            dw = (1 / n_samples) * np.dot(features.T, (probabilities - labels))
            db = (1 / n_samples) * np.sum(probabilities - labels)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss {loss:.4f}")

    def predict_proba(self, features):
        linear_output = np.dot(features, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, features):
        return (self.predict_proba(features) >= 0.5).astype(int)

    def evaluate(self, y_true, y_pred):
        true_positive = np.sum((y_true == 1) & (y_pred == 1))
        false_positive = np.sum((y_true == 0) & (y_pred == 1))
        false_negative = np.sum((y_true == 1) & (y_pred == 0))

        precision = true_positive / (true_positive + false_positive + 1e-9)
        recall = true_positive / (true_positive + false_negative + 1e-9)
        f1_score = 2 * precision * recall / (precision + recall + 1e-9)

        return precision, recall, f1_score

    def plot_loss(self):
        plt.plot(self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.show()
