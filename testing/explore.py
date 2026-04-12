import os

DATA_DIR = r"../data"

classes = sorted(os.listdir(DATA_DIR))
total = 0

print(f"{'Class':<10} {'Images':>8}")
print("-" * 20)
for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"{cls:<10} {len(imgs):>8}")
    total += len(imgs)

print("-" * 20)
print(f"{'TOTAL':<10} {total:>8}")
print(f"{'Classes':<10} {len(classes):>8}")