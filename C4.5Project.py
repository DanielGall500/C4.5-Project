import numpy as np
import pandas as pd
from sklearn import tree

osx_filename = '\Users\dannyg\Dropbox\Datasets\Breast Cancer Data\breast-cancer-wisconsin.csv'
windows_filename = 'C:/Users/dano/Dropbox/Datasets/Breast Cancer Data/breast-cancer-wisconsin.csv'

pre_data = pd.read_csv(windows_filename, sep=';')
print len(pre_data)

cols_list = ['id', 'thickness', 'uni_size', 'uni_shape', 'marginal_adhesion', 'epithelial_size', \
             'bare_nuclei', 'chromatin', 'norm_nucleoli', 'mitoses', 'class']

data = pre_data[pre_data['bare_nuclei'] != '?']

clf = tree.DecisionTreeClassifier()

clf.fit()