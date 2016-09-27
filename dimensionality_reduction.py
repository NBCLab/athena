# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
Perform dimensionality reduction of naive bag of words feature

"""
from sklearn.feature_selection import SelectKBest, chi2
import os
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns

# Set figure style
sns.set_style("darkgrid")

# Load files
data_dir = "/home/data/nbc/athena/athena-data/"
features_dir = os.path.join(data_dir, "features/")
labels_dir = os.path.join(data_dir, "labels/")
figures_dir = os.path.join(data_dir, "figures/")

train_features_file = os.path.join(features_dir, "train_nbow.csv")
test_features_file = os.path.join(features_dir, "test_nbow.csv")
train_labels_file = os.path.join(labels_dir, "train.csv")
test_labels_file = os.path.join(labels_dir, "test.csv")

out_train_features_file = os.path.join(features_dir, "train_nbow_reduced.csv")
out_test_features_file = os.path.join(features_dir, "test_nbow_reduced.csv")

figure_file = os.path.join(figures_dir, "selected_feature_counts.png")

train_features_df = pd.read_csv(train_features_file, index_col="pmid")
train_features = train_features_df.as_matrix()
test_features_df = pd.read_csv(test_features_file, index_col="pmid")
test_features = test_features_df.as_matrix()

train_labels_df = pd.read_csv(train_labels_file, index_col="pmid")
train_labels = train_labels_df.as_matrix()

# Perform labelwise feature selection
clf = Pipeline([("fs", SelectKBest(chi2, k=100)), ("svm", LinearSVC())])
multi_clf = OneVsRestClassifier(clf)
multi_clf.fit(train_features, train_labels)
selected_features = []
for i, output in enumerate(multi_clf.estimators_):
    selected_features += list(output.named_steps["fs"].get_support(indices=True))

# Combine selected features across labels and reduce to features that were
# useful for five or more labels.
s = pd.Series(selected_features)
vc = s.value_counts()
vc_red = vc[vc>4]

keep_features = sorted(list(vc_red.index))
column_names = train_features_df.columns.values
selected_columns = column_names[keep_features]
print len(keep_features)

# Plot and save count figure
vc.index = range(vc.shape[0])
ax = vc.plot(kind="line")
fig = ax.get_figure()
fig.savefig(figure_file)

# Create reduced features dataframes for both test and train
out_train_features_df = train_features_df[selected_columns]
out_test_features_df = test_features_df[selected_columns]
out_train_features_df.to_csv(out_train_features_file)
out_test_features_df.to_csv(out_test_features_file)
