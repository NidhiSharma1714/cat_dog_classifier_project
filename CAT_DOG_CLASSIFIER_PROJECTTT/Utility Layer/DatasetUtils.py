import os
import shutil
import random

# Original dataset paths
cat_dir = r"C:\Users\admin\Downloads\dataSet\PetImages\Cat"
dog_dir = r"C:\Users\admin\Downloads\dataSet\PetImages\Dog"

# Target directories
base_dir = os.path.join(os.getcwd(), "dataset")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")
test_dir = os.path.join(base_dir, "test")

# Create directories with Correct Names ✅
for d in [train_dir, val_dir, test_dir]:
    for cls in ["Cat", "Dog"]:  # <-- Capitalized & Singular
        os.makedirs(os.path.join(d, cls), exist_ok=True)

# Parameters
split_train = 0.7
split_val = 0.15
split_test = 0.15

# Collect all files
all_cats = [f for f in os.listdir(cat_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
all_dogs = [f for f in os.listdir(dog_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

random.shuffle(all_cats)
random.shuffle(all_dogs)

def split_and_copy(files, source_dir, class_name):
    total = len(files)
    train_split = int(split_train * total)
    val_split = int((split_train + split_val) * total)

    for i, fname in enumerate(files):
        src = os.path.join(source_dir, fname)
        if i < train_split:
            dst = os.path.join(train_dir, class_name, fname)
        elif i < val_split:
            dst = os.path.join(val_dir, class_name, fname)
        else:
            dst = os.path.join(test_dir, class_name, fname)
        shutil.copyfile(src, dst)

# ✅ Apply with Correct Folder Names
split_and_copy(all_cats, cat_dir, "Cat")
split_and_copy(all_dogs, dog_dir, "Dog")

print("✅ Dataset prepared successfully!")
