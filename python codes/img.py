import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import pickle

input_dir = r"c:\Users\Arman\Desktop\IMAGE CLASSIFIER\clf-data"
categories = ["empty", "not_empty"]

data = []
labels = []

for category_idx,category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)
        img = resize(img, (32,32))
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test , y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)



#train classifier
classifier = SVC()

parameters = [{'gamma': [0.1,0.01,0.001], 'C': [10,100,1000]}]

grid_serach = GridSearchCV(classifier, parameters)

grid_serach.fit(x_train, y_train)

#testing

best_estimator = grid_serach.best_estimator_
accuracy = grid_serach.best_score_
print(accuracy)

for mean, params in zip(grid_serach.cv_results_["mean_test_score"],
                        grid_serach.cv_results_["params"]):
    print(mean,params)

print("\nbest:")
print(grid_serach.best_score_, grid_serach.best_params_)

y_predict = best_estimator.predict(x_test)

pickle.dump(best_estimator, open("./model.p", "wb"))