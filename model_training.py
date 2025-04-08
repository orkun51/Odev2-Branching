import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
"""

# **🔍 0. GPU Kullanımı Kontrol Ediliyor**
print("\n🔍 TensorFlow GPU Kontrol Ediliyor...")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU Kullanılıyor: {gpus[0].name}\n")
    except RuntimeError as e:
        print(f"⚠️ GPU Bellek Ayarlarken Hata: {e}")
else:
    print("\n⚠️ GPU bulunamadı, CPU ile devam edilecek.")
"""
# **⏳ 1. CSV Dosyasını Yükleme ve Kontroller**
start_time = time.time()
print("\n📂 CSV Dosyası Yükleniyor...")
csv_file = "Cizim_Etiketleri_Encoded.csv"

try:
    df = pd.read_csv(csv_file)
    print(f"✅ CSV başarıyla yüklendi! {len(df)} satır bulundu.")
except FileNotFoundError:
    print(f"❌ Hata: {csv_file} dosyası bulunamadı!")
    exit()

# **Sütunları Kontrol Et ve Gerekirse Düzelt**
print("\n📌 Mevcut Sütunlar:", df.columns)
if "Kategori_Kodu" not in df.columns and "Kategori" in df.columns:
    label_encoder = LabelEncoder()
    df["Kategori_Kodu"] = label_encoder.fit_transform(df["Kategori"])
    print("✅ Kategori sütunu başarıyla sayısal hale getirildi!")

df["Kategori_Kodu"] = df["Kategori_Kodu"].astype(int)

if df["Kategori_Kodu"].isnull().sum() > 0:
    print("⚠️ Hata: Kategori_Kodu sütununda eksik değerler var!")
    exit()

labels = df["Kategori_Kodu"].values
num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)
print(f"✅ Labels başarıyla işlendi! Toplam {num_classes} farklı sınıf bulundu.")

# **⏳ 2. Görselleri Yükleme ve Ön İşleme**
print("\n🖼️ Görseller Yükleniyor...")
image_folder = "Etiketlenen_Cizimler/"
img_size = (224, 224)
image_data = []
valid_files = []

for index, row in df.iterrows():
    file_path = os.path.join(image_folder, row["Dosya_Adi"])
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        image_data.append(img_array)
        valid_files.append(row["Kategori_Kodu"])
    except Exception as e:
        print(f"⚠️ Hata: {file_path} - {e}")

image_data = np.array(image_data)
labels = np.array(valid_files)
labels = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
print(f"✅ {len(image_data)} görsel başarıyla işlendi ve eğitim/test verisi oluşturuldu.")

# **⏳ 3. Transfer Learning Modeli (ResNet50)**
print("\n🔄 ResNet50 Modeli Yükleniyor...")
model_load_start = time.time()

try:
    base_model = ResNet50(weights='/Users/orkun/Desktop/AutomatedEvaluation/Cizim Degerlendirme/model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    print("✅ ResNet50 modeli başarıyla yüklendi!")
except Exception as e:
    print(f"❌ ResNet50 modeli yüklenirken hata oluştu: {e}")
    exit()

model_load_end = time.time()
print(f"🔄 Model yükleme süresi: {model_load_end - model_load_start:.2f} saniye")

# **Modeli İnşa Et**
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# **⏳ 4. Modelin Derlenmesi**
print("\n🔍 Model Derleniyor...")
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model derleme tamamlandı!")

# Modelin özetini yazdır
model.summary()

# **⏳ 5. Modeli Eğitme ve Takip Etme**
checkpoint_filepath = "./best_model_weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1
)

print("\n🚀 Model Eğitimi Başlıyor...\n")
train_start = time.time()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint_callback]
)

train_end = time.time()
print(f"✅ Model eğitimi tamamlandı! ({train_end - train_start:.2f} saniye)")

# **⏳ 6. Model Performansını Görselleştirme**
print("\n📊 Eğitim Sonuçları Görselleştiriliyor...")
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title("Eğitim ve Doğrulama Doğruluğu")
plt.show()

# **⏳ 7. Modeli Kaydetme**
print("\n💾 Model Kaydediliyor...")
model.save("transfer_learning_model.h5")
print("✅ Model başarıyla kaydedildi!")

# **Toplam süreyi göster**
end_time = time.time()
print(f"\n🎉 Tüm işlem tamamlandı! Toplam süre: {end_time - start_time:.2f} saniye")