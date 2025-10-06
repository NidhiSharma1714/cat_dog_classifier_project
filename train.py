import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Clean corrupted images
# ----------------------------
def remove_corrupted_images(folder_path):
    removed_files = 0
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(root, fname)
                try:
                    img = Image.open(fpath)
                    img.verify()  # check if image can be opened
                except (IOError, SyntaxError) as e:
                    print(f"⚠️ Removing corrupted image: {fpath}")
                    os.remove(fpath)
                    removed_files += 1
    print(f"✅ Done! Removed {removed_files} corrupted images.")

dataset_dir = os.path.join(os.getcwd(), "dataset")
remove_corrupted_images(dataset_dir)

# ----------------------------
# 2️⃣ Dataset paths
# ----------------------------
train_dir = os.path.join(dataset_dir, "train")
val_dir = os.path.join(dataset_dir, "validation")

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20

# ----------------------------
# 3️⃣ Data Generators with augmentation
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ----------------------------
# Check class balance
# ----------------------------
def print_class_balance(generator):
    classes_count = {cls: 0 for cls in generator.class_indices}
    for _, labels in generator:
        for i, cls_name in enumerate(generator.class_indices):
            classes_count[cls_name] += sum(labels == i)
        break  # just check first batch
    print("Class indices:", generator.class_indices)
    print("Class balance in first batch:", classes_count)

print_class_balance(train_data)
print_class_balance(val_data)

# ----------------------------
# 4️⃣ Lightweight CNN Model
# ----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ----------------------------
# 5️⃣ Callbacks & Training
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Validation accuracy
val_loss, val_acc = model.evaluate(val_data)
print(f"Validation Accuracy: {val_acc:.2%}")

# ----------------------------
# 6️⃣ Save Model
# ----------------------------
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/cat_dog_classifier.h5")
print("✅ Lightweight model trained and saved!")

# ----------------------------
# 7️⃣ Plot Accuracy & Loss
# ----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Loss")
plt.legend()

plt.show()
