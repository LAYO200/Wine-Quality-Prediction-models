# metrics, plots, confusion/residuals
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import OUTPUT_DIR


def plot_nn_learning_curves(history_cls, history_reg, output_dir=OUTPUT_DIR):
    """Plot 1 & 2: NN learning curves (same code as your script)."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: classification NN
    plt.figure()
    epochs_cls = range(1, len(history_cls["train_acc"]) + 1)
    plt.plot(epochs_cls, history_cls["train_acc"], label="Train accuracy")
    plt.plot(epochs_cls, history_cls["val_acc"], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Plot 1 — NN Classification Learning Curve")
    plt.legend()
    plt.tight_layout()
    plot1_path = os.path.join(
        output_dir, "plot1_nn_classification_learning_curve.png"
    )
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.show()

    # Plot 2: regression NN
    plt.figure()
    epochs_reg = range(1, len(history_reg["train_loss"]) + 1)
    plt.plot(epochs_reg, history_reg["train_loss"], label="Train MSE")
    plt.plot(epochs_reg, history_reg["val_loss"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Plot 2 — NN Regression Learning Curve")
    plt.legend()
    plt.tight_layout()
    plot2_path = os.path.join(
        output_dir, "plot2_nn_regression_learning_curve.png"
    )
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.show()

    return plot1_path, plot2_path


def plot_confusion_matrix_best(
    best_final_cls_model,
    best_final_cls_type,
    best_final_cls_name,
    X_te,
    X_te_nn,
    y_cls_te,
    output_dir=OUTPUT_DIR,
):
    #Plot 3: confusion matrix for best final classifier.
    if best_final_cls_type == "nn":
        y_pred_cls_test = best_final_cls_model.predict(X_te_nn)
    else:
        y_pred_cls_test = best_final_cls_model.predict(X_te)

    cm = confusion_matrix(y_cls_te, y_pred_cls_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot()
    plt.title(f"Plot 3 — Confusion Matrix (Test) — {best_final_cls_name}")
    plt.tight_layout()
    plot3_path = os.path.join(
        output_dir, "plot3_best_classification_confusion_matrix.png"
    )
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.show()
    return plot3_path

# Plot 4: residuals vs predicted for best final regressor.
def plot_residuals_best(
    best_final_reg_model,
    best_final_reg_type,
    best_final_reg_name,
    X_te,
    X_te_nn,
    y_reg_te,
    output_dir=OUTPUT_DIR,
):

    if best_final_reg_type == "nn":
        y_pred_reg_test = best_final_reg_model.predict(X_te_nn)
    else:
        y_pred_reg_test = best_final_reg_model.predict(X_te)

    residuals = y_reg_te - y_pred_reg_test

    plt.figure()
    plt.scatter(y_pred_reg_test, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Quality (Test)")
    plt.ylabel("Residuals (y - y_pred)")
    plt.title(f"Plot 4 — Residuals vs Predicted (Test) — {best_final_reg_name}")
    plt.tight_layout()
    plot4_path = os.path.join(
        output_dir, "plot4_best_regression_residuals_vs_predicted.png"
    )
    plt.savefig(plot4_path, dpi=200, bbox_inches="tight")
    plt.show()
    return plot4_path

# Plot 5: Decision Tree feature importances.
def plot_feature_importances(tree_cls, X, output_dir=OUTPUT_DIR):

    tree_model = tree_cls.named_steps["clf"]
    importances = tree_model.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": X.columns, "importance": importances}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(6, 4))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Plot 5 — Feature Importances (Decision Tree Classifier)")
    plt.tight_layout()
    plot5_path = os.path.join(
        output_dir, "plot5_feature_importances_decision_tree_classifier.png"
    )
    plt.savefig(plot5_path, dpi=200, bbox_inches="tight")
    plt.show()
    return plot5_path


def make_final_tables(
    best_classical_cls_name,
    best_classical_cls_metrics,
    best_classical_reg_name,
    best_classical_reg_metrics,
    m_nn_cls,
    m_nn_reg,
    output_dir=OUTPUT_DIR,
):
    #Tables 1 & 2
    # Table 1 – Classification comparison: best classical vs NN
    table1_cls = pd.DataFrame(
        [
            [
                best_classical_cls_name,
                best_classical_cls_metrics["val_accuracy"],
                best_classical_cls_metrics["val_f1"],
                best_classical_cls_metrics["test_accuracy"],
                best_classical_cls_metrics["test_f1"],
            ],
            [
                "MLPClassifier",
                m_nn_cls["val_accuracy"],
                m_nn_cls["val_f1"],
                m_nn_cls["test_accuracy"],
                m_nn_cls["test_f1"],
            ],
        ],
        columns=["Model", "Val_Accuracy", "Val_F1", "Test_Accuracy", "Test_F1"],
    )

    # Table 2 – Regression comparison: best classical vs NN
    table2_reg = pd.DataFrame(
        [
            [
                best_classical_reg_name,
                best_classical_reg_metrics["val_mae"],
                best_classical_reg_metrics["val_rmse"],
                best_classical_reg_metrics["test_mae"],
                best_classical_reg_metrics["test_rmse"],
            ],
            [
                "MLPRegressor",
                m_nn_reg["val_mae"],
                m_nn_reg["val_rmse"],
                m_nn_reg["test_mae"],
                m_nn_reg["test_rmse"],
            ],
        ],
        columns=["Model", "Val_MAE", "Val_RMSE", "Test_MAE", "Test_RMSE"],
    )

    print("\n=== Table 1 — Classification: Classical vs NN (Val/Test) ===")
    print(table1_cls.to_string(index=False))

    print("\n=== Table 2 — Regression: Classical vs NN (Val/Test) ===")
    print(table2_reg.to_string(index=False))

    table1_path = os.path.join(
        output_dir, "table1_classification_comparison_classical_vs_nn.csv"
    )
    table2_path = os.path.join(
        output_dir, "table2_regression_comparison_classical_vs_nn.csv"
    )

    table1_cls.to_csv(table1_path, index=False)
    table2_reg.to_csv(table2_path, index=False)

    return table1_path, table2_path
