import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import joblib
import sklearn
import sklearn.svm as svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.family'] = 'Heiti TC'

data_root = r'D:\Work_Dr\Coffee_bean\Source'
classes = os.listdir(data_root)

specs = []
labels = []

for cls in classes:
    cls_path = os.path.join(data_root, cls)
    img_names = sorted(os.listdir(cls_path))
    # Save MSI (Multiple Spectrum Image)
    msi = []
    for name in img_names:
        img_path = os.path.join(cls_path, name)
        print(img_path)
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        # Blend 4 channels into MSI
        msi.append(img)
    # Make sequence correspond to OpenCV standard
    msi = np.transpose(np.array(msi), (1, 2, 0))
    print(msi.shape)
    # Use channel 4 for segmentation
    im_seg = msi[:, :, 3]
    im_seg = (im_seg / im_seg.max() * 255).astype(np.uint8)     # idk
    mask = cv.threshold(im_seg, 100, 255, cv.THRESH_BINARY)[1]
    plt.figure()
    plt.imshow(mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        target_mask = np.zeros_like(im_seg)
        perimeter = cv.arcLength(cont, True)
        if perimeter < 50:
            # Skip if area is too small
            continue
        cv.drawContours(target_mask, [cont], 0, (255, 255, 255), -1)
        plt.figure()
        plt.imshow(target_mask)
        target_mask = target_mask.astype(bool)
        spec = np.mean(msi[target_mask], axis=(0))
        specs.append(spec)
        labels.append(cls)
        # break

specs = np.array(specs)
labels = np.array(labels)
print(specs.shape, labels.shape)

def median_filter(kernel_size = 5):
    # Pre-check
    if not isinstance(kernel_size, int) or kernel_size <= 1 or kernel_size % 2 == 0:
        print("Kernel size must be an odd number.")
        return
    process_count = 0
    for cls in classes:
        dir_in = os.path.join(data_root, cls)
        # Create output directory
        dir_out = dir_in + '_MF'
        os.makedirs(dir_out, exist_ok=True)
        img_names = sorted(os.listdir(dir_in))
        for name in img_names:
            img_path = os.path.join(dir_in, name)
            # Read & Process
            img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
            img_filtered = cv.medianBlur(img, kernel_size)
            process_count += 1
            # Output
            output_path = os.path.join(dir_out, name)
            success = cv.imwrite(output_path, img_filtered)
            if success:
                print(f"Image saved successfully. {process_count}")
            else:
                print(f"Failure! {process_count}")

# 建模分析 使用 GridSearch 因为较少的数据量
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def grid_search(X_train, y_train, savePath='./svm2'):

    svm_pipe = Pipeline([
        ('scalar', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', svm.LinearSVC()),
    ])

    svm_clf = GridSearchCV(svm_pipe, param_grid={'pca__n_components': [20]})


