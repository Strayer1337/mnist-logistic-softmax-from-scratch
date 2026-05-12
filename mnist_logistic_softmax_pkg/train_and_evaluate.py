from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from mnist_classification.data_loader import (
    load_mnist_images,
    load_mnist_labels,
    preprocess_binary_data,
    preprocess_multiclass_data,
    TRAIN_IMAGES_PATH,
    TRAIN_LABELS_PATH,
    TEST_IMAGES_PATH,
    TEST_LABELS_PATH,
)
from mnist_classification.logistic_regression import LogisticRegression
from mnist_classification.softmax_regression import SoftmaxRegression


def main():
    # Load raw data
    train_images = load_mnist_images(TRAIN_IMAGES_PATH, 60000)
    train_labels = load_mnist_labels(TRAIN_LABELS_PATH)
    test_images = load_mnist_images(TEST_IMAGES_PATH, 10000)
    test_labels = load_mnist_labels(TEST_LABELS_PATH)

    # --- Assignment 1: Custom Logistic Regression (binary: chữ số 0 vs 1) ---
    train_images_lr, train_labels_lr = preprocess_binary_data(train_images, train_labels)
    test_images_lr, test_labels_lr = preprocess_binary_data(test_images, test_labels)

    lr_model = LogisticRegression()
    lr_model.fit(train_images_lr, train_labels_lr)

    lr_predictions = lr_model.predict(test_images_lr)
    precision, recall, f1 = lr_model.evaluate(test_labels_lr, lr_predictions)
    print(f"\n[Custom Logistic Regression]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    lr_model.plot_loss()

    # --- Assignment 2: Custom Softmax Regression (multiclass: 10 chữ số) ---
    train_images_sr, train_labels_sr = preprocess_multiclass_data(train_images, train_labels)
    test_images_sr, test_labels_sr = preprocess_multiclass_data(test_images, test_labels)

    softmax_model = SoftmaxRegression()
    softmax_model.fit(train_images_sr, train_labels_sr)

    softmax_predictions = softmax_model.predict(test_images_sr)
    precision, recall, f1 = softmax_model.evaluate(test_labels_sr, softmax_predictions)
    print(f"\n[Custom Softmax Regression]")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    softmax_model.plot_loss()

    # --- Assignment 3: Sklearn Logistic Regression (binary) ---
    sklearn_lr = SklearnLR(max_iter=1000)
    sklearn_lr.fit(train_images_lr, train_labels_lr)
    lr_sklearn_preds = sklearn_lr.predict(test_images_lr)

    print(f"\n[Sklearn Logistic Regression - Binary]")
    print("Confusion matrix:")
    print(confusion_matrix(test_labels_lr, lr_sklearn_preds))
    print(f"Precision: {precision_score(test_labels_lr, lr_sklearn_preds):.4f}")
    print(f"Recall:    {recall_score(test_labels_lr, lr_sklearn_preds):.4f}")
    print(f"F1 Score:  {f1_score(test_labels_lr, lr_sklearn_preds):.4f}")

    # --- Assignment 3: Sklearn Softmax Regression (multiclass) ---
    sklearn_softmax = SklearnLR(solver="lbfgs", max_iter=1000)
    sklearn_softmax.fit(train_images_sr, train_labels_sr)
    softmax_sklearn_preds = sklearn_softmax.predict(test_images_sr)

    print(f"\n[Sklearn Softmax Regression - Multiclass]")
    print("Confusion matrix:")
    print(confusion_matrix(test_labels_sr, softmax_sklearn_preds))
    print(f"Precision: {precision_score(test_labels_sr, softmax_sklearn_preds, average='macro'):.4f}")
    print(f"Recall:    {recall_score(test_labels_sr, softmax_sklearn_preds, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(test_labels_sr, softmax_sklearn_preds, average='macro'):.4f}")


if __name__ == "__main__":
    main()
