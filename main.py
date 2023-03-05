import os
from imutils import paths
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from sklearn.svm import SVC

'''
    numPoints：選取中心像素周圍的像素點的個數
    radius：選取的區域的半徑

    method有以下的選擇
    method : {‘default’, ‘ror’, ‘uniform’, ‘var’}

    ‘default’: 特徵值受到旋轉影響，一旦旋轉失效.
    ‘ror’: 特徵值就不受旋轉影響
    ‘uniform’:改進了具有均勻模式的旋轉不變性和灰度和旋轉不變的角度空間的更精細量化。這個最好用。
    ‘var’: 旋轉但不是灰度不變
'''

training_data_path = 'dataset/training'
test_data_path = 'dataset/test'

numPoints = 24
radius = 8

train_data = []
train_labels = []
for image_path in paths.list_images(training_data_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 我們這裡用 uniform的方法來萃取特徵是為了避免受到物體旋轉的影響
    lbp_feathures = feature.local_binary_pattern(gray, numPoints, radius, method='uniform')
    # 分門別類 分24組
    hist, _ = np.histogram(lbp_feathures.ravel(), bins=np.arange(0, numPoints))
    # 調整一下資料型態，下面才有辦法算除法
    hist = hist.astype('float')
    # 縮小差異性
    hist /= hist.sum()

    train_data.append(hist)
    # os.path.sep就是根據系統不同，代表路徑不同的分隔符號
    train_labels.append(image_path.split(os.path.sep)[-2])

# C越大越準確 但可能會出現過儗合現象
# svm https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E6%94%AF%E6%92%90%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-svm-%E8%A9%B3%E7%B4%B0%E6%8E%A8%E5%B0%8E-c320098a3d2e
model = SVC(kernel='poly', C=100, random_state=1, gamma='scale')
model.fit(train_data, train_labels)

plt.figure(figsize=(15, 10))
for i,image_path in enumerate(paths.list_images(test_data_path)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract LBPH features
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, numPoints))
    hist = hist.astype("float")

    hist /= (hist.sum())

    test_label = image_path.split(os.path.sep)[-2]
    # 要進模型
    prediction = model.predict(hist.reshape(1, -1))

    # display the image and the prediction
    color = (0,255,0) if test_label==prediction[0] else (0,0,255)
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    plt.subplot(2,3,i+1)
    plt.axis('off')
    plt.title(test_label)
    plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

plt.savefig('result.jpg')
plt.show()
