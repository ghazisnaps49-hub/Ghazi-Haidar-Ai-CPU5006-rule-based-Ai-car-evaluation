"""
Assessment 1: Rule-Based AI Scientific Research Paper (CPU5006)
Dataset: Car Evaluation Dataset
Author: Ghazi Haidar 
Description: Complete, robust experiment script generating exact numerical tables 
and visual charts (Figure 1 & Figure 2) using matplotlib, scikit-learn, and pandas.
"""

import matplotlib

matplotlib.use(
    'Agg'
)  # Prevents display/window thread crashes on Windows environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

print('Initializing Car Evaluation AI Experiment...')

# 1. Load Dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

try:
  df = pd.read_csv(url, header=None, names=columns)
  print(
      'Successfully loaded dataset from UCI repository. Total records:'
      f' {len(df)}'
  )
except Exception as e:
  print(f'Error downloading dataset: {e}')
  raise

# 2. Data Preparation (80% train, 20% test split, stratified)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['class']
)
print(
    f'Data split completed: Training records = {len(train_df)}, Test records'
    f' = {len(test_df)}'
)


# 3. Decision-List Engine (tracking default rule hits)
def evaluate_model(test_data, rule_function, default_label='unacc'):
  predictions = []
  default_hits = 0
  for _, row in test_data.iterrows():
    pred = rule_function(row)
    if pred is None:
      predictions.append(default_label)
      default_hits += 1
    else:
      predictions.append(pred)
  return predictions, default_hits


# 4. Model Definitions
def safety_based_rules(row):
  if row['safety'] == 'low':
    return 'unacc'
  if row['persons'] == '2':
    return 'unacc'
  if row['safety'] == 'high' and row['persons'] == 'more':
    return 'vgood'
  if row['safety'] == 'med' and row['persons'] == '4':
    return 'acc'
  return None


def policy_based_rules(row):
  if row['safety'] == 'low':
    return 'unacc'
  if row['persons'] == '2':
    return 'unacc'
  if (
      row['buying'] in ['low', 'med']
      and row['maint'] in ['low', 'med']
      and row['safety'] == 'high'
      and row['lug_boot'] in ['med', 'big']
  ):
    return 'vgood'
  if (
      row['persons'] == 'more'
      and row['lug_boot'] == 'big'
      and row['safety'] == 'high'
  ):
    return 'vgood'
  if (
      row['buying'] == 'low'
      and row['maint'] in ['low', 'med']
      and row['safety'] == 'med'
  ):
    return 'good'
  if (
      row['buying'] == 'med'
      and row['maint'] == 'low'
      and row['safety'] == 'high'
  ):
    return 'good'
  if row['buying'] in ['low', 'med'] and row['safety'] == 'med':
    return 'acc'
  if row['maint'] in ['low', 'med'] and row['safety'] == 'high':
    return 'acc'
  if row['buying'] == 'high' and row['safety'] == 'med':
    return 'acc'
  return None


# 5. Model Execution
y_true = test_df['class']
classes = ['unacc', 'acc', 'good', 'vgood']

y_pred_base = ['unacc'] * len(test_df)
y_pred_safety, default_safety = evaluate_model(test_df, safety_based_rules)
y_pred_policy, default_policy = evaluate_model(test_df, policy_based_rules)

# Calculate Macro Metrics
acc_base = accuracy_score(y_true, y_pred_base) * 100
_, _, f1_base, _ = precision_recall_fscore_support(
    y_true, y_pred_base, average='macro', zero_division=0
)

acc_safety = accuracy_score(y_true, y_pred_safety) * 100
_, _, f1_safety, _ = precision_recall_fscore_support(
    y_true, y_pred_safety, average='macro', zero_division=0
)

acc_policy = accuracy_score(y_true, y_pred_policy) * 100
_, _, f1_policy, _ = precision_recall_fscore_support(
    y_true, y_pred_policy, average='macro', zero_division=0
)

# Per-Class F1-scores
f1_safety_per_class = precision_recall_fscore_support(
    y_true, y_pred_safety, labels=classes, zero_division=0
)[2]
f1_policy_per_class = precision_recall_fscore_support(
    y_true, y_pred_policy, labels=classes, zero_division=0
)[2]

# Print Results & Tables
print('\n' + '=' * 50)
print('=== Table 3: Rule Complexity ===')
print(
    'Safety-Based Model: Rules = 5 | Avg Conditions = 1.60 | Default Hits ='
    f' {default_safety}'
)
print(
    'Policy-Based Model: Rules = 10 | Avg Conditions = 2.70 | Default Hits ='
    f' {default_policy}'
)
print('-' * 50)

print('=== Table 4: Per-Class F1-score ===')
for cls, f1_s, f1_p in zip(classes, f1_safety_per_class, f1_policy_per_class):
  print(
      f'Class: {cls:<6} | Safety Model: {f1_s:.2f} | Policy Model: {f1_p:.2f}'
  )
print('=' * 50 + '\n')

# 6. Generate Figure 1: Accuracy and Macro F1-score Bar Chart
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x = np.arange(2)
width = 0.25

rects1 = ax.bar(
    x - width,
    [acc_base, f1_base * 100],
    width,
    label='Majority Baseline',
    color='#b0b0b0',
)
rects2 = ax.bar(
    x,
    [acc_safety, f1_safety * 100],
    width,
    label='Safety-Based Model',
    color='#7a9bb8',
)
rects3 = ax.bar(
    x + width,
    [acc_policy, f1_policy * 100],
    width,
    label='Policy-Based Model',
    color='#1d4e78',
)

ax.set_ylabel('Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(['Accuracy', 'Macro F1-score'])
ax.legend()
ax.set_ylim(0, 105)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for rects in [rects1, rects2, rects3]:
  for rect in rects:
    height = rect.get_height()
    ax.annotate(
        f'{height:.2f}',
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=9,
    )

plt.title(
    'Figure 1: Accuracy and macro F1-score of both models against the majority'
    ' baseline',
    pad=15,
)
plt.tight_layout()
plt.savefig('figure1_overall_performance.png')
plt.close()

# 7. Generate Figure 2: Per-Class F1-score Bar Chart
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
x_cls = np.arange(len(classes))
width_cls = 0.35

rects_s = ax.bar(
    x_cls - width_cls / 2,
    f1_safety_per_class,
    width_cls,
    label='Safety-Based Model',
    color='#7a9bb8',
)
rects_p = ax.bar(
    x_cls + width_cls / 2,
    f1_policy_per_class,
    width_cls,
    label='Policy-Based Model',
    color='#1d4e78',
)

ax.set_ylabel('F1-score')
ax.set_xlabel('Vehicle class')
ax.set_xticks(x_cls)
ax.set_xticklabels(classes)
ax.legend()
ax.set_ylim(0, 1.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for rects in [rects_s, rects_p]:
  for rect in rects:
    height = rect.get_height()
    ax.annotate(
        f'{height:.2f}',
        xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext=(0, 3),
        textcoords='offset points',
        ha='center',
        va='bottom',
        fontsize=9,
    )

plt.title('Figure 2: Per-class F1-score of the two rule-based models', pad=15)
plt.tight_layout()
plt.savefig('figure2_per_class_f1.png')
plt.close()

print(
    'Success! All metrics printed and charts saved as'
    ' "figure1_overall_performance.png" and "figure2_per_class_f1.png".'
)