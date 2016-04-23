import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

osx_filename = '\Users\dannyg\Dropbox\Datasets\Breast Cancer Data\breast-cancer-wisconsin.csv'
windows_filename = 'C:/Users/dano/Dropbox/Datasets/Breast Cancer Data/breast-cancer-wisconsin.csv'

pre_data = pd.read_csv(windows_filename, sep=';')

features = ['id', 'thickness', 'uni_size', 'uni_shape', 'marginal_adhesion', 'epithelial_size', \
            'bare_nuclei', 'chromatin', 'norm_nucleoli', 'mitoses']
target = ['class']

data = pre_data[pre_data['bare_nuclei'] != '?']

x_train, x_test, y_train, y_test \
    = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

print 'Accuracy Score', accuracy_score(y_test, predictions)


