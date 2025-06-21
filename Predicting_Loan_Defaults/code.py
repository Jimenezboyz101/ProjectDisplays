import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

sns.set()
np.random.seed(416)

loans = pd.read_csv('lending_data.csv')
loans.head()

grade_order = sorted(loans['grade'].unique())
sns.countplot(x='grade', data=loans, order=grade_order)

ownership_order = sorted(loans['home_ownership'].unique())
sns.countplot(x='home_ownership', data=loans, order=ownership_order)

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop(columns='bad_loans')

only_safe = loans[loans['safe_loans'] == 1]
only_bad = loans[loans['safe_loans'] == -1]

print(f'Number safe  loans: {len(only_safe)} ({len(only_safe) * 100.0 / len(loans):.2f}%)')
print(f'Number risky loans: {len(only_bad)} ({len(only_bad) * 100.0 / len(loans):.2f}%)')

mode_grade = loans['grade'].value_counts().idxmax()
percent_rent = (loans['home_ownership'] == 'RENT').mean()

features = [
    'grade', 'sub_grade', 'short_emp', 'emp_length_num', 'home_ownership',
    'dti', 'purpose', 'term', 'last_delinq_none', 'last_major_derog_none',
    'revol_util', 'total_rec_late_fee'
]
target = 'safe_loans'

loans = loans[features + [target]]
loans.head()

loans.columns
loans = pd.get_dummies(loans)
features = list(loans.columns)
features.remove('safe_loans')
features
loans.head()

from sklearn.model_selection import train_test_split
train_data, validation_data = train_test_split(loans, test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
X_train = pd.get_dummies(train_data[features])
y_train = train_data[target]

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model.fit(X_train, y_train)

import graphviz
from sklearn import tree

def draw_tree(tree_model, features):
    tree_data = tree.export_graphviz(tree_model, impurity=False, feature_names=features,
                                     class_names=tree_model.classes_.astype(str), filled=True, out_file=None)
    graph = graphviz.Source(tree_data)
    display(graph)

small_tree_model = DecisionTreeClassifier(max_depth=2, random_state=0)
small_tree_model.fit(train_data[features], train_data[target])
draw_tree(small_tree_model, features)

X_validation = pd.get_dummies(validation_data[features])
X_validation = X_validation.reindex(columns=X_train.columns, fill_value=0)
y_validation = validation_data[target]

train_preds = decision_tree_model.predict(X_train)
validation_preds = decision_tree_model.predict(X_validation)

decision_train_accuracy = (train_preds == y_train).mean()
decision_validation_accuracy = (validation_preds == y_validation).mean()

big_tree_model = DecisionTreeClassifier(max_depth=10)
big_tree_model.fit(X_train, y_train)

big_train_preds = big_tree_model.predict(X_train)
big_validation_preds = big_tree_model.predict(X_validation)

big_train_accuracy = (big_train_preds == y_train).mean()
big_validation_accuracy = (big_validation_preds == y_validation).mean()

from sklearn.model_selection import GridSearchCV
hyperparameters = {'min_samples_leaf': [1, 10, 50, 100, 200, 300], 'max_depth': [1, 5, 10, 15, 20]}
base_tree = DecisionTreeClassifier()

search = GridSearchCV(estimator=base_tree, param_grid=hyperparameters, cv=6, return_train_score=True)
search.fit(X_train, y_train)
print(search.best_params_)

def plot_scores(ax, title, search, hyperparameters, score_key):
    cv_results = search.cv_results_
    scores = cv_results[score_key]
    scores = scores.reshape((len(hyperparameters['max_depth']), len(hyperparameters['min_samples_leaf'])))
    max_depths = cv_results['param_max_depth'].reshape(scores.shape).data.astype(int)
    min_samples_leafs = cv_results['param_min_samples_leaf'].reshape(scores.shape).data.astype(int)
    ax.plot_wireframe(max_depths, min_samples_leafs, scores)
    ax.view_init(20, 220)
    ax.set_xlabel('Maximum Depth')
    ax.set_ylabel('Minimum Samples Leaf')
    ax.set_zlabel('Accuracy')
    ax.set_title(title)

fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
plot_scores(ax1, 'Train Accuracy', search, hyperparameters, 'mean_train_score')
plot_scores(ax2, 'Validation Accuracy', search, hyperparameters, 'mean_test_score')

import scipy.stats

class RandomForest416:
    def __init__(self, num_trees, max_depth=1):
        self._trees = [DecisionTreeClassifier(max_depth=max_depth) for i in range(num_trees)]

    def fit(self, X, y):
        n = len(X)
        for tree in self._trees:
            bootstrap_indices = np.random.randint(0, n, size=n)
            X_bootstrap = X.iloc[bootstrap_indices]
            y_bootstrap = y.iloc[bootstrap_indices]
            tree.fit(X_bootstrap, y_bootstrap)

    def predict(self, X):
        predictions = np.zeros((len(X), len(self._trees)))
        for i, tree in enumerate(self._trees):
            preds = tree.predict(X)
            predictions[:, i] = preds
        return scipy.stats.mode(predictions, axis=1, keepdims=False).mode

rf_model = RandomForest416(num_trees=2, max_depth=1)
rf_model.fit(X_train, y_train)

rf_train_preds = rf_model.predict(X_train)
rf_validation_preds = rf_model.predict(X_validation)

rf_train_accuracy = (rf_train_preds == y_train).mean()
rf_validation_accuracy = (rf_validation_preds == y_validation).mean()

from sklearn.metrics import accuracy_score
depths = list(range(1, 26, 2))
dt_accuracies = []
rf_accuracies = []

for i in depths:
    print(f'Depth {i}')
    tree = DecisionTreeClassifier(max_depth=i)
    tree.fit(train_data[features], train_data[target])
    dt_accuracies.append((accuracy_score(tree.predict(train_data[features]), train_data[target]),
                          accuracy_score(tree.predict(validation_data[features]), validation_data[target])))

    rf = RandomForest416(15, max_depth=i)
    rf.fit(train_data[features], train_data[target])
    rf_accuracies.append((accuracy_score(rf.predict(train_data[features]), train_data[target]),
                          accuracy_score(rf.predict(validation_data[features]), validation_data[target])))

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(depths, [acc[0] for acc in dt_accuracies], label='DecisionTree')
axs[0].plot(depths, [acc[0] for acc in rf_accuracies], label='RandomForest416')
axs[1].plot(depths, [acc[1] for acc in dt_accuracies], label='DecisionTree')
axs[1].plot(depths, [acc[1] for acc in rf_accuracies], label='RandomForest416')
axs[0].set_title('Train Data')
axs[1].set_title('Validation Data')
for ax in axs:
    ax.legend()
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')