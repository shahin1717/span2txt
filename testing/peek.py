import os
import cv2
import matplotlib.pyplot as plt

DATA_DIR = "../data_cleaned"

classes = sorted(os.listdir(DATA_DIR))
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
axes = axes.flatten()

for i, cls in enumerate(classes):
    cls_path = os.path.join(DATA_DIR, cls)
    img_name = os.listdir(cls_path)[0]
    img = cv2.imread(os.path.join(cls_path, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(cls, fontsize=14)
    axes[i].axis("off")

# hide empty subplots
for j in range(len(classes), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig("samples.png")
print("Saved → samples.png")