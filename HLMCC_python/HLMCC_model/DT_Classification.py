import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from collections import Counter

# Set random seed for reproducibility
np.random.seed(10)

# Load the data
clustered_data = pd.read_csv("/mnt/c/Users/tjudd/OneDrive - Umich/ML4CYB/IDSL/HLMCC_code/HLMCC_python/Datasets/clustered_Dataset/satellite-unsupervised-lablled.csv")

# Check the balance of the classes
print(Counter(clustered_data['ClusterLabel']))  # "o" = "anomaly" class and "n" = "normal"

# Remove the ID column
clustered_data.drop(clustered_data.columns[0], axis=1, inplace=True)

# Convert the label to categorical
clustered_data['ClusterLabel'] = clustered_data['ClusterLabel'].astype('category')

# Calculate the oversampling ratio to be within the defined limits
majority_class_count = clustered_data['ClusterLabel'].value_counts().max()
minority_class_count = clustered_data['ClusterLabel'].value_counts().min()
over_ratio = (0.1 * majority_class_count - minority_class_count) / minority_class_count 
over_ratio = min(over_ratio, 1.0)

print(f'Sampling Strategy: {over_ratio}')

# Use SMOTE for oversampling
smote = SMOTE(sampling_strategy=over_ratio, random_state=10)
X = clustered_data.drop('ClusterLabel', axis=1)
y = clustered_data['ClusterLabel']
X_smote, y_smote = smote.fit_resample(X, y)

# Check the result
print(Counter(y_smote))

# Evaluation metrics and seeds
fpr = []
fm = []
pri = []
pr_c = []
sensitivity = []
seeds = [10, 100, 1000, 2000, 3500]

# Train the model 5 times
for seed in seeds:
    np.random.seed(seed)
    
    # Shuffle and split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=seed, stratify=y_smote)

    # Train the decision tree
    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_train, y_train)

    # Predict for the test data
    y_pred = clf.predict(X_test)

    # Create confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, labels=y.cat.categories)
    print(conf_mat)

    # Evaluation metrics
    class_report = classification_report(y_test, y_pred, labels=y.cat.categories, output_dict=True)
    
    fm.append(class_report['macro avg']['f1-score'])
    pri.append(class_report['macro avg']['precision'])
    sensitivity.append(class_report['macro avg']['recall'])

    # Compute False Positive Rate
    tn, fp, fn, tp = conf_mat.ravel()
    if (fp + tn) > 0:
        fpr.append(fp / (fp + tn))
    else:
        fpr.append(np.nan)

    # Compute the area under the precision-recall curve (AUCPR)
    y_pred_proba = clf.predict_proba(X_test)
    
    # Taking the probability of the positive class ("o" = "anomaly" class)
    y_scores_anomaly = y_pred_proba[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test == "o", y_scores_anomaly, pos_label=True)
    pr_auc_score = auc(recall, precision)
    pr_c.append(pr_auc_score)

# The mean of results
print(f"False Positive Rate : Mean {np.nanmean(fpr)}")
print(f"False Positive Rate : SD {np.nanstd(fpr, ddof=1)}")

print(f"F-measure : {np.nanmean(fm)}")
print(f"F-measure : SD {np.nanstd(fm, ddof=1)}")

print(f"Sensitivity : {np.nanmean(sensitivity)}")
print(f"Sensitivity : SD {np.nanstd(sensitivity, ddof=1)}")

print(f"Precision : {np.nanmean(pri)}")
print(f"Precision : SD {np.nanstd(pri, ddof=1)}")

print(f"The area under the precision-recall curve (AUCPR) : {np.nanmean(pr_c)}")
print(f"The area under the precision-recall curve (AUCPR) : SD {np.nanstd(pr_c, ddof=1)}")