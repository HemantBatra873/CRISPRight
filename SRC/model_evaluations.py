
def evaluate_model(model, X_test, y_test, threshold=0.5):

    import numpy as np

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, average_precision_score
    )

    print(f"\n\n\nEVALUATING MODEL: {model.name}")

    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = recall

    fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred_proba)

    print("\n\nCLASSIFICATION METRICS\n")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"AUC-ROC:      {auc_roc:.4f}")
    print(f"AUC-PR:       {auc_pr:.4f}")
    print(f"Sensitivity:  {sensitivity:.4f}")
    print(f"Specificity:  {specificity:.4f}")

    print("\n\nCONFUSION MATRIX\n")
    print(f"                  Predicted")
    print(f"                  Neg    Pos")
    print(f"Actual  Neg     {tn:>5}  {fp:>5}")
    print(f"        Pos     {fn:>5}  {tp:>5}")

    print("\n\nDETAILED METRICS\n")
    print(f"True Positives:   {tp}")
    print(f"True Negatives:   {tn}")
    print(f"False Positives:  {fp}")
    print(f"False Negatives:  {fn}")
    print(f"Total Samples:    {len(y_test)}")

    print("\n\nCLASSIFICATION REPORT\n")
    print(classification_report(y_test, y_pred,
                                target_names=['Off-target', 'On-target'],
                                digits=4))

    results = {
        'model_name': model.name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds_roc},
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=['Off-target', 'On-target'],
            output_dict=True
        )
    }

    return results

def plot_evaluation_results(results):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.metrics import precision_recall_curve
    import warnings
    warnings.filterwarnings("ignore")

    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (16, 10)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12

    model_name = results.get("model_name", "Model")

    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_pred_proba = results["y_pred_proba"]
    cm = results["confusion_matrix"]
    fpr = results["roc_curve"]["fpr"]
    tpr = results["roc_curve"]["tpr"]
    auc_roc = results["auc_roc"]
    auc_pr = results["auc_pr"]
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

    metrics = {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1-Score": results["f1_score"],
        "AUC-ROC": results["auc_roc"],
        "AUC-PR": results["auc_pr"],
        "Sensitivity": results["sensitivity"],
        "Specificity": results["specificity"],
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Evaluation Summary for {model_name}", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {auc_pr:.3f}")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Off-target", "On-target"],
                yticklabels=["Off-target", "On-target"],
                cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax = axes[1, 0]
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    sns.barplot(x=metric_names, y=metric_values, hue=metric_names, palette="viridis", legend=False, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Metrics")
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    for i, v in enumerate(metric_values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    ax = axes[1, 1]
    sns.histplot(y_pred_proba[y_true == 0], bins=30, color="skyblue", label="Off-target", kde=True, ax=ax)
    sns.histplot(y_pred_proba[y_true == 1], bins=30, color="orange", label="On-target", kde=True, ax=ax)
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Predicted Probability (y_pred_proba)")
    ax.set_ylabel("Count")
    ax.legend()

    ax = axes[1, 2]
    report = results["classification_report"]
    report_df = (
        sns.heatmap(
            np.array([
                [report["Off-target"]["precision"], report["Off-target"]["recall"], report["Off-target"]["f1-score"]],
                [report["On-target"]["precision"], report["On-target"]["recall"], report["On-target"]["f1-score"]],
            ]),
            annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=["Precision", "Recall", "F1-score"],
            yticklabels=["Off-target", "On-target"],
            cbar=False, ax=ax
        )
    )
    ax.set_title("Class-wise Metrics (from classification report)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print(f"\n\nPlots generated for {model_name}")
