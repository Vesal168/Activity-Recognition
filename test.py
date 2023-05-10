
import numpy as np
from prepare_data import *

features_person_1 = np.loadtxt(features_file_person_1, delimiter=",")
features_all = np.concatenate([features_person_1])
features_person_1_2_3 = np.concatenate([features_person_1])

print('\nTest Classifiers - learn: 1,2,3 test: 1')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_person_1_2_3, features_person_1))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_person_1_2_3, features_person_1))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_person_1_2_3, features_person_1))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_person_1_2_3, features_person_1))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_person_1_2_3, features_person_1))


