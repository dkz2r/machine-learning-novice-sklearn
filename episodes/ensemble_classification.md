## Stacking: classification
import seaborn as sns
penguins = sns.load_dataset('penguins')

feature_names = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
penguins.dropna(subset=feature_names, inplace=True)

species_names = penguins['species'].unique()

# Define data and targets
X = penguins[feature_names]

y = penguins.species

# Split data in training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(f'train size: {X_train.shape}')
print(f'test size: {X_test.shape}')

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier

# training estimators 
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=1, random_state=5)
gb_clf = GradientBoostingClassifier(random_state=5)
gp_clf = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=5)
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=5)

voting_reg = VotingClassifier([("rf", rf_clf), ("gb", gb_clf), ("gp", gp_clf), ("dt", dt_clf)])

# fit voting estimator
voting_reg.fit(X_train, y_train)

# lets also train the individual models for comparison
rf_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)
gp_clf.fit(X_train, y_train)
dt_clf.fit(X_train, y_train)

import matplotlib.pyplot as plt

# make predictions
X_test_20 = X_test[:20] # first 20 for visualisation

rf_pred = rf_clf.predict(X_test_20)
gb_pred = gb_clf.predict(X_test_20)
gp_pred = gp_clf.predict(X_test_20)
dt_pred = dt_clf.predict(X_test_20)
voting_pred = voting_reg.predict(X_test_20)

print(rf_pred)
print(gb_pred)
print(gp_pred)
print(dt_pred)
print(voting_pred)

plt.figure()
plt.plot(gb_pred,  "o", color="green", label="GradientBoostingClassifier")
plt.plot(rf_pred,  "o", color="blue", label="RandomForestClassifier")
plt.plot(gp_pred,  "o", color="darkblue", label="GuassianProcessClassifier")
plt.plot(dt_pred,  "o", color="lightblue", label="DecisionTreeClassifier")
plt.plot(voting_pred,  "x", color="red", ms=10, label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()

print(f'random forest: {rf_clf.score(X_test, y_test)}')

print(f'gradient boost: {gb_clf.score(X_test, y_test)}')

print(f'guassian process: {gp_clf.score(X_test, y_test)}')

print(f'decision tree: {dt_clf.score(X_test, y_test)}')

print(f'voting regressor: {voting_reg.score(X_test, y_test)}')