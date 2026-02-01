import os
import numpy as np
import cv2 as cv
import cv2
import joblib
import json
from matplotlib import pyplot as plt
import sklearn.svm as svm
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.utils import shuffle
import glob
import sklearn
from gx_spectral.feature import roi, spectrum
import spectral

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


# data_x.tofile('aa.npy')

from gx_spectral.visualization import drawer
# Visualise spectrum distribution
spec_pos = [450, 555, 660, 850]
spec_pos = np.array(spec_pos, dtype=str)
img = drawer.show_specs_class(spec_pos, specs, labels, '反射率')
cv2.imwrite('Plots/Reflective_rate.jpg', img[0])
cv2.imwrite('Plots/Average_spectrum.jpg', img[1])
img = drawer.show_dimension_reduction(specs, labels)
cv2.imwrite('Plots/image3.jpg', img)


# 建模分析
from sklearn.pipeline import Pipeline

def grid_search(X_train, y_train, savePath='./svm2'):
    tuned_parameters = [
        {"kernel": ["rbf", 'linear'], "gamma": 1 / np.power(10, np.arange(5)), "C": np.power(10, np.arange(5))}, ]
    param_grid = {
        'pca__n_components': np.arange(2, 5),
        'clf__C': np.power(10, np.arange(5)),
        'clf__gamma': 1 / np.power(10, np.arange(5)),
        'clf__kernel': ["rbf", 'linear']
    }
    score = 'accuracy'
    # score = 'neg_mean_squared_error'
    #     scaler = StandardScaler().fit(X_train)
    #     X_scaled = scaler.transform(X_train)
    pipeline = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=2)), ('clf', svm.SVC())])
    clf = GridSearchCV(pipeline, param_grid, scoring=score, cv=5)

    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("Best score:", clf.best_score_)
    # 保存最优参数
    paraFilePath = os.path.join(savePath, 'best_model_parameters.json')
    #     with open(paraFilePath, 'w+') as f:
    #         json.dump(clf.best_params_, f)

    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    return clf.best_estimator_



# 采用10折交叉验证进行建模的评估
from sklearn.model_selection import StratifiedKFold ,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from gx_spectral.visualization import drawer
import pandas as pd
from gx_spectral.preprocess import spectrum as gxspectrum
from sklearn.decomposition import PCA
from sklearn import preprocessing
# pca = PCA(n_components=3)
X = specs
y = labels
strKFold = StratifiedKFold(n_splits=5, shuffle=False)
model = make_pipeline(
    StandardScaler(),
    # PCA(n_components=5),
#                       RandomForestClassifier(),
                      svm.SVC(kernel='linear', C=1000, gamma=0.001)
                     )

# 采用自动划分数据集的方式进行建模的评估
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9, stratify=y)
# model = grid_search(X_train, y_train)
scores = cross_val_score(model,X,y,cv=strKFold)
print(scores)
print(scores.mean())


print(model)
model.fit(X_train,y_train)
score_res = model.score(X_train,y_train)
print("The train score of model is : %f"%score_res)
score_res = model.score(X_test,y_test)
print("The test score of model is : %f"%score_res)
y_pred = model.predict(X_test)
img, report = drawer.show_confusion_matrix(y_test, y_pred)
cv2.imwrite('Plots/prediction.png', img)
# 将 report 保存成本地文件
pd.DataFrame(report).transpose().to_csv("report.csv", index= True, float_format='%.2f')