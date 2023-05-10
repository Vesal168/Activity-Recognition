import mgr.data.readrawdata as rd
import mgr.calc.signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

""" PERSON_1 - prepare data - start """
STANDING_PERSON_1 = rd.read('mgr/data/resources/person_1/standing.csv')
STANDING_PERSON_1['magnitude'] = sig.magnitude(STANDING_PERSON_1)

WALKING_PERSON_1 = rd.read('mgr/data/resources/person_1/walking.csv')
WALKING_PERSON_1['magnitude'] = sig.magnitude(WALKING_PERSON_1)

DOWNSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/downstairs.csv')
DOWNSTAIRS_PERSON_1['magnitude'] = sig.magnitude(DOWNSTAIRS_PERSON_1)

UPSTAIRS_PERSON_1 = rd.read('mgr/data/resources/person_1/upstairs.csv')
UPSTAIRS_PERSON_1['magnitude'] = sig.magnitude(UPSTAIRS_PERSON_1)

RUNNING_PERSON_1 = rd.read('mgr/data/resources/person_1/running.csv')
RUNNING_PERSON_1['magnitude'] = sig.magnitude(RUNNING_PERSON_1)

""" PERSON_1 - prepare data - stop """

activities_person_1 = [STANDING_PERSON_1, WALKING_PERSON_1, DOWNSTAIRS_PERSON_1, UPSTAIRS_PERSON_1, RUNNING_PERSON_1]
output_file_path_person_1 = 'mgr/data/features/person_1/Features.csv'

with open(output_file_path_person_1, 'w') as features_file:
    rows = csv.writer(features_file)
    for i in range(0, len(activities_person_1)):
        for f in sig.extract_features(activities_person_1[i]):
            rows.writerow([i] + f)

features_person_1 = np.loadtxt('mgr/data/features/person_1/Features.csv', delimiter=",")

print('\nTest Classifiers on PERSON_1 learned with data collected by PERSON_1')
print('K-Neighbors Classifier  ', sig.test_knn_cls_one_set(features_person_1))
print('Decision Tree Classifier', sig.test_decision_tree_cls_one_set(features_person_1))
print('Random Forest Classifier', sig.test_random_forest_cls_one_set(features_person_1))
print('MLP Classifier          ', sig.test_mlp_cls_one_set(features_person_1))
print('GaussianNB              ', sig.test_gaussian_nb_cls_one_set(features_person_1))


features_all = np.concatenate([features_person_1])
k_neighbors_cls_all = KNeighborsClassifier()
decision_tree_cls_all = DecisionTreeClassifier()
random_forest_cls_all = RandomForestClassifier()
mlp_cls_all = MLPClassifier()
gaussian_nb_cls_all = GaussianNB()


print('\nTest Classifiers on PERSON_1 learned with all data')
print('K-Neighbors Classifier  ', sig.test_and_learn_knn_cls(features_all, features_person_1))
print('Decision Tree Classifier', sig.test_and_learn_decision_tree_cls(features_all, features_person_1))
print('Random Forest Classifier', sig.test_and_learn_random_forest_cls(features_all, features_person_1))
print('MLP Classifier          ', sig.test_and_learn_mlp_cls(features_all, features_person_1))
print('GaussianNB              ', sig.test_and_learn_gaussian_nb_cls(features_all, features_person_1))

