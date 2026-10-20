# UTS-Semester-7
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Konfigurasi Awal & Reproduksibilitas
print("TensorFlow Version:", tf.__version__)
print("GPU Aktif:", tf.config.list_physical_devices("GPU"))
keras.utils.set_random_seed(42)

# Dictionary untuk menyimpan rekapitulasi hasil Kasus 1 - 3
rekap_hasil = {}

# ==============================================================================
# STUDI KASUS 1: CNN UNTUK KLASIFIKASI FOTO PRODUK FASHION
# ==============================================================================
print("\n" + "="*50)
print(" STUDI KASUS 1: CNN vs FCNN (Fashion-MNIST)")
print("="*50)

# 1. Memuat Dataset
try:
    (x_train_img, y_train_img), (x_test_img, y_test_img) = keras.datasets.fashion_mnist.load_data()
except Exception as e:
    print("Gagal mengunduh Fashion-MNIST bawaan, membuat dataset sintetis untuk pengujian...")
    x_train_img = np.random.randint(0, 256, (6000, 28, 28), dtype=np.uint8)
    y_train_img = np.random.randint(0, 10, (6000,), dtype=np.uint8)
    x_test_img = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
    y_test_img = np.random.randint(0, 10, (1000,), dtype=np.uint8)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 2. Pra-pemrosesan Data
x_train_img = x_train_img.astype("float32") / 255.0
x_test_img = x_test_img.astype("float32") / 255.0
x_train_img = np.expand_dims(x_train_img, -1)
x_test_img = np.expand_dims(x_test_img, -1)

# Helper Training
def compile_and_train(model, x_tr, y_tr, epochs=5):
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    t0 = time.time()
    hist = model.fit(x_tr, y_tr, epochs=epochs, batch_size=128, validation_split=0.1, verbose=2)
    sec_per_epoch = (time.time() - t0) / epochs
    return hist, sec_per_epoch

# --- Baseline: FCNN ---
fcnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
], name="fcnn")

hist_fcnn, time_fcnn = compile_and_train(fcnn, x_train_img, y_train_img, epochs=5)
loss_fcnn, acc_fcnn = fcnn.evaluate(x_test_img, y_test_img, verbose=0)
rekap_hasil["FCNN"] = {"Akurasi": acc_fcnn, "Param": fcnn.count_params(), "Waktu/Epoch": time_fcnn, "Catatan": "Baseline FCNN"}

# --- Model CNN ---
cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu", name="conv1"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", name="conv2"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
], name="cnn")

hist_cnn, time_cnn = compile_and_train(cnn, x_train_img, y_train_img, epochs=5)
loss_cnn, acc_cnn = cnn.evaluate(x_test_img, y_test_img, verbose=0)
rekap_hasil["CNN"] = {"Akurasi": acc_cnn, "Param": cnn.count_params(), "Waktu/Epoch": time_cnn, "Catatan": "Ekstraksi fitur spasial baik"}

print(f"FCNN -> Akurasi Uji: {acc_fcnn:.4f} | Param: {fcnn.count_params():,}")
print(f"CNN  -> Akurasi Uji: {acc_cnn:.4f} | Param: {cnn.count_params():,}")

# Visualisasi Feature Map CNN
feature_model = keras.Model(inputs=cnn.inputs, outputs=cnn.get_layer("conv1").output)
fmap = feature_model.predict(x_test_img[0:1], verbose=0)

plt.figure(figsize=(10, 2))
plt.subplot(1, 10, 1)
plt.imshow(x_test_img[0].squeeze(), cmap="gray")
plt.axis("off")
plt.title("Input")
for n in range(9):
    plt.subplot(1, 10, n + 2)
    plt.imshow(fmap[0, :, :, n], cmap="viridis")
    plt.axis("off")
    plt.title(f"F{n}")
plt.suptitle("Feature Map Conv1 (9 Filter Pertama)")
plt.show()

# ==============================================================================
# STUDI KASUS 2: RNN, LSTM, DAN GRU UNTUK ANALISIS SENTIMEN
# ==============================================================================
print("\n" + "="*50)
print(" STUDI KASUS 2: SimpleRNN vs LSTM vs GRU (IMDB)")
print("="*50)

max_features = 10000
maxlen = 200

try:
    (xr_train, yr_train), (xr_test, yr_test) = keras.datasets.imdb.load_data(num_words=max_features)
except Exception as e:
    print("Gagal mengunduh IMDB bawaan, membuat dataset sintetis teks...")
    xr_train = [list(np.random.randint(4, max_features, size=np.random.randint(50, 150))) for _ in range(2000)]
    yr_train = np.random.randint(0, 2, size=(2000,))
    xr_test = [list(np.random.randint(4, max_features, size=np.random.randint(50, 150))) for _ in range(500)]
    yr_test = np.random.randint(0, 2, size=(500,))

xr_train = keras.utils.pad_sequences(xr_train, maxlen=maxlen)
xr_test = keras.utils.pad_sequences(xr_test, maxlen=maxlen)

def build_rnn_model(cell="lstm", units=64, embed_dim=64):
    rnn_layers = {
        "simple": layers.SimpleRNN,
        "lstm": layers.LSTM,
        "gru": layers.GRU
    }
    model = keras.Sequential([
        layers.Input(shape=(maxlen,)),
        layers.Embedding(max_features, embed_dim),
        rnn_layers[cell](units),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ], name=f"model_{cell}")

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

EPOCHS = 3
cells = ["simple", "lstm", "gru"]

for cell in cells:
    print(f"\n--- Training Model {cell.upper()} ---")
    model = build_rnn_model(cell=cell, units=64, embed_dim=64)
    t0 = time.time()
    model.fit(xr_train, yr_train, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=2)
    time_per_epoch = (time.time() - t0) / EPOCHS

    loss, acc = model.evaluate(xr_test, yr_test, verbose=0)
    rekap_hasil[cell.upper()] = {
        "Akurasi": acc,
        "Param": model.count_params(),
        "Waktu/Epoch": time_per_epoch,
        "Catatan": f"Model RNN dengan cell {cell.upper()}"
    }
    print(f"{cell.upper()} -> Akurasi Uji: {acc:.4f} | Param: {model.count_params():,}")

# ==============================================================================
# REKAPITULASI HASIL AKHIR
# ==============================================================================
print("\n" + "="*50)
print(" REKAPITULASI PERFORMA MODEL")
print("="*50)
print(f"{'Model':<12} | {'Akurasi':<10} | {'Parameter':<12} | {'Waktu/Epoch (s)':<16} | Catatan")
print("-"*85)
for k, v in rekap_hasil.items():
    print(f"{k:<12} | {v['Akurasi']:<10.4f} | {v['Param']:<12,} | {v['Waktu/Epoch']:<16.2f} | {v['Catatan']}")
