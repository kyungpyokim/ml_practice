import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)


def show_confusion_matrix(y_test, pred):
  cm = confusion_matrix(y_test, pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Confusion Matrix')
  plt.show()
  
def print_evaluation(y_test, pred, pred_proba):
  print(f"정확도: {accuracy_score(y_test, pred):.4f}")
  print(f"정밀도: {precision_score(y_test, pred):.4f}")
  print(f"재현율: {recall_score(y_test, pred):.4f}")
  print(f"오차행렬: {confusion_matrix(y_test, pred)}")
  print(f"F1 Score: {f1_score(y_test, pred):.4f}")
  print(f"ROC-AUC: {roc_auc_score(y_test, pred_proba):.4f}")
