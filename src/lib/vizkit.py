import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def show_confusion_matrix(y_test, pred):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def print_evaluation(y_test, pred, pred_proba):
    print(f"정확도: {accuracy_score(y_test, pred):.4f}")
    print(f"정밀도: {precision_score(y_test, pred):.4f}")
    print(f"재현율: {recall_score(y_test, pred):.4f}")
    print(f"오차행렬: {confusion_matrix(y_test, pred)}")
    print(f"F1 Score: {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, pred_proba):.4f}")


def evaluate_regression_model(y_test, y_pred):
    """
    회귀 모델의 주요 성능 지표(MAE, RMSE, R2)를 출력합니다.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("📊 Regression Model Evaluation")
    print("-------------------------------")
    print(f"MAE  (Average Error) : {mae:.4f}")
    print(f"MSE  (Average Error) : {mse:.4f}")
    print(f"RMSE (Large Error)   : {rmse:.4f}")
    print(f"R2 Score (Fit)      : {r2:.4f}")
    print("-------------------------------")


def show_importances(model, features):
    # 1. 학습된 모델(예: rf_reg)에서 중요도 추출
    # 'final_features'는 앞서 상관계수 0.1 이상으로 추출한 컬럼 리스트입니다.
    importances = model.feature_importances_
    indices = np.argsort(importances)  # 중요도 순으로 정렬

    # 2. 시각화 데이터 구성
    feature_importances = pd.DataFrame(
        {
            "Feature": [features[i] for i in indices],
            "Importance": importances[indices],
        }
    )

    # 3. 가로 막대 그래프 시각화
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=feature_importances,
        x="Importance",
        y="Feature",
        hue="Feature",  # y축 변수를 hue에 할당
        palette="magma",
        legend=False,  # 불필요한 범례 제거
    )
    plt.title("Feature Importances for Delivery Time Prediction", fontsize=15)
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


def compare_models(models, X_train, y_train, X_test, y_test):
    results = []
    best_pred = None
    max_r2 = 0

    # 2. 반복문을 통한 모델 학습 및 평가
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        if max_r2 < r2:
            max_r2 = r2
            best_pred = preds

        results.append({"Model": name, "MAE": mae, "R2 Score": r2})

    # 3. 결과를 데이터프레임으로 변환
    df_results = pd.DataFrame(results)

    # 4. 시각화 (MAE와 R2 Score)
    _, ax1 = plt.subplots(figsize=(10, 6))

    # MAE 막대 그래프 (낮을수록 좋음)
    sns.barplot(
        data=df_results, x="Model", y="MAE", hue="MAE", ax=ax1, palette="Blues_d"
    )
    ax1.set_title("Model Performance Comparison", fontsize=15)
    ax1.set_ylabel("MAE (Lower is Better)")

    # R2 Score 선 그래프 (높을수록 좋음)
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_results, x="Model", y="R2 Score", marker="o", color="red", ax=ax2
    )
    ax2.set_ylabel("R2 Score (Higher is Better)")

    plt.show()

    return best_pred


def best_model(y_test, pred):
    # 가장 성능이 좋은 모델의 예측값으로 시각화
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, pred, alpha=0.3, color="blue")
    plt.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )  # 기준선

    plt.xlabel("Actual Delivery Time (min)")
    plt.ylabel("Predicted Delivery Time (min)")
    plt.title("Actual vs Predicted")
    plt.show()
