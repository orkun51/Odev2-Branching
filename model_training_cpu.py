import os
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

# **✅ Gereksiz GPU loglarını kapat ve sadece CPU kullan**
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Metal loglarını engelle
tf.config.set_visible_devices([], 'GPU')  # GPU'yu devre dışı bırak

print("\n✅ GPU tamamen devre dışı bırakıldı, sadece CPU kullanılacak!")

# **1. CSV Dosyasını Yükleme ve Kontroller**
csv_file = "Cizim_Etiketleri_Encoded.csv"
df = pd.read_csv(csv_file)
print("📂 CSV başarıyla yüklendi!")

# **Sütunları Kontrol Et ve Gerekirse Düzelt**
print("📌 Mevcut Sütunlar:", df.columns)

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
print(f"✅ Labels başarıyla işlendi! Toplam {num_classes} sınıf bulundu.")

# **2. Görselleri Yükleme ve Ön İşleme**
print("\n🖼️ Görseller Yükleniyor...")
image_folder = "Etiketlenen_Cizimler/"
img_size = (224, 224)
image_data = []

for index, row in df.iterrows():
    file_path = os.path.join(image_folder, row["Dosya_Adi"])
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize(img_size)
        img_array = np.array(img) / 255.0
        image_data.append(img_array)
    except Exception as e:
        print(f"⚠️ Hata: {file_path} - {e}")

image_data = np.array(image_data)
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
print(f"✅ {len(image_data)} görsel başarıyla işlendi!")

# **3. Transfer Learning Modeli (ResNet50)**
print("\n🔄 ResNet50 Modeli Yükleniyor...")
base_model = ResNet50(weights='model_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# **4. Modelin Derlenmesi**
print("\n🔍 Model Derleniyor...")
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
print("✅ Model derleme tamamlandı!")

# **5. Modeli Eğitme**
print("\n🚀 Model eğitimi başlıyor...")
history = model.fit(X_train, y_train, epochs=5, batch_size=8, validation_data=(X_test, y_test))
print("✅ Model eğitimi tamamlandı!")

# **6. Model Performansını Görselleştirme**
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title("Eğitim ve Doğrulama Doğruluğu")
plt.show()

# **7. Modeli Kaydetme**
print("\n💾 Model Kaydediliyor...")
model.save("transfer_learning_model.h5")
print("✅ Model başarıyla kaydedildi!")