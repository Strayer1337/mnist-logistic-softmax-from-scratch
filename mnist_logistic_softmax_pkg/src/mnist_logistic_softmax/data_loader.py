import numpy as np

TRAIN_IMAGES_PATH = "./Data/train-images.idx3-ubyte"
TRAIN_LABELS_PATH = "./Data/train-labels.idx1-ubyte"
TEST_IMAGES_PATH = "./Data/t10k-images.idx3-ubyte"
TEST_LABELS_PATH = "./Data/t10k-labels.idx1-ubyte"


def load_mnist_images(path, n_images, img_shape=(28, 28)):
    with open(path, "rb") as f:
        f.read(16)  # skip header
        image_bytes = f.read()
    return np.frombuffer(image_bytes, dtype=np.uint8).reshape(n_images, *img_shape)


def load_mnist_labels(path):
    with open(path, "rb") as f:
        f.read(8)  # skip header
        label_bytes = f.read()
    return np.frombuffer(label_bytes, dtype=np.uint8)


def preprocess_binary_data(images, labels):
    """Flatten, normalize và chỉ giữ nhãn 0 và 1."""
    flattened = images.reshape(images.shape[0], -1) / 255.0
    mask = (labels == 0) | (labels == 1)
    return flattened[mask], labels[mask]


def preprocess_multiclass_data(images, labels):
    """Flatten và normalize toàn bộ 10 chữ số."""
    flattened = images.reshape(images.shape[0], -1) / 255.0
    return flattened, labels


if __name__ == "__main__":
    train_images = load_mnist_images(TRAIN_IMAGES_PATH, 60000)
    train_labels = load_mnist_labels(TRAIN_LABELS_PATH)
    test_images = load_mnist_images(TEST_IMAGES_PATH, 10000)
    test_labels = load_mnist_labels(TEST_LABELS_PATH)

    print(f"train images: {train_images.shape}, train labels: {train_labels.shape}")
    print(f"test images: {test_images.shape}, test labels: {test_labels.shape}")
