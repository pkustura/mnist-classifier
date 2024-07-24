# Standard library
import struct
import numpy as np

def load_data(labels_path, images_path):
    """
    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels
    """
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def structured_load_np():
    # training data
    tr_i, tr_l = load_data("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte")
    tr_i = tr_i.astype(np.float64) / 255.0  # Normalize pixel values to [0, 1]
    tr_l = [vectorized_result(y) for y in tr_l]

    te_i, te_l = load_data("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte")
    te_i = te_i.astype(np.float64) / 255.0  # Normalize pixel values to [0, 1]
    # te_l = [vectorized_result(y) for y in te_l]

    # Reshape images to be column vectors
    tr_i = [img.reshape(-1, 1) for img in tr_i]
    te_i = [img.reshape(-1, 1) for img in te_i]

    print(f"Training data shape: {tr_i[0].shape}, Label shape: {tr_l[0].shape}")
    print(f"Test data shape: {te_i[0].shape}, Label shape: {te_l[0].shape}")

    return (list(zip(tr_i, tr_l)), list(zip(te_i, te_l)))
        
    

def structured_load_torch():
    # training data
    tr_i, tr_l = load_data("./data/train-labels.idx1-ubyte", "./data/train-images.idx3-ubyte")
    tr_i = tr_i.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    te_i, te_l = load_data("./data/t10k-labels.idx1-ubyte", "./data/t10k-images.idx3-ubyte")
    te_i = te_i.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]

    # Reshape images to be 2D arrays (each row is an image, 784 pixels) 
    tr_i = tr_i.reshape(-1, 784)
    te_i = te_i.reshape(-1, 784)

    print(f"Training data shape: {tr_i[0].shape}, Label shape: {tr_l[0].shape}")
    print(f"Test data shape: {te_i[0].shape}, Label shape: {te_l[0].shape}")

    return (tr_i, tr_l), (te_i, te_l)


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
