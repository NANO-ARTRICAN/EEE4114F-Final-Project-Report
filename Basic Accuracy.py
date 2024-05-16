import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_validate
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import joblib 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
data = pd.read_csv("pixel_values2.0.csv")
data = shuffle(data)
data
X = data.drop(["Folder"], axis=1)
Y =data["Folder"]
idx = 542
img = X.loc[idx].values.reshape(28,28)
print(Y[idx])
plt.imshow(img)
train_x,test_x,train_y,test_y = train_test_split(X,Y, test_size = 0.2)
classifier=SVC(kernel="linear", random_state=6)
classifier.fit(train_x,train_y)
joblib.dump(classifier, "digit_recogniser")
prediction = classifier.predict(test_x)
print("Accuracy = ", metrics.accuracy_score(prediction, test_y))