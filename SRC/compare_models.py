

def compare_models(results):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("\n\n\nMODEL COMPARISON ANALYSIS")

    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'AUC-ROC': result['auc_roc'],
            'AUC-PR': result['auc_pr'],
            'Sensitivity': result['sensitivity'],
            'Specificity': result['specificity'],
            'TP': result['true_positives'],
            'TN': result['true_negatives'],
            'FP': result['false_positives'],
            'FN': result['false_negatives']
        })

    df_comparison = pd.DataFrame(comparison_data)

    print("\nPERFORMANCE METRICS SUMMARY")
    print(df_comparison.to_string(index=False))

    print("\nBEST PERFORMERS BY METRIC")

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC',
               'AUC-PR', 'Sensitivity', 'Specificity']

    for metric in metrics:
        best_idx = df_comparison[metric].idxmax()
        best_model = df_comparison.loc[best_idx, 'Model']
        best_value = df_comparison.loc[best_idx, metric]
        print(f"{metric:15s}: {best_model:20s} ({best_value:.4f})")

    print("\n\nOVERALL MODEL RANKING")

    metrics_to_rank = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    df_normalized = df_comparison[metrics_to_rank].copy()

    df_comparison['Overall_Score'] = df_normalized.mean(axis=1)
    df_ranked = df_comparison.sort_values('Overall_Score', ascending=False)

    print("\nRank Model Overall Score")
    for idx, (_, row) in enumerate(df_ranked.iterrows(), 1):
        print(f"{idx:2d}.   {row['Model']:20s}     {row['Overall_Score']:.4f}")

    print("\nSTATISTICAL SUMMARY")

    stats_df = df_comparison[metrics].describe().T
    stats_df = stats_df[['mean', 'std', 'min', 'max']]
    print(stats_df.to_string())

    print("\nCONFUSION MATRIX ANALYSIS\n")

    for model_name, result in results.items():
        cm = result['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp

        print(f"\n{model_name}:")
        print(f"  True Negatives:  {tn:5d} ({tn/total*100:5.2f}%)")
        print(f"  False Positives: {fp:5d} ({fp/total*100:5.2f}%)")
        print(f"  False Negatives: {fn:5d} ({fn/total*100:5.2f}%)")
        print(f"  True Positives:  {tp:5d} ({tp/total*100:5.2f}%)")

    print("\nGENERATING COMPARISON VISUALIZATIONS")

    fig = plt.figure(figsize=(20, 12))
    ax1 = plt.subplot(2, 3, 1, projection='polar')

    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    colors_radar = plt.cm.Set3(np.linspace(0, 1, len(results)))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['accuracy'], result['precision'], result['recall'],
                  result['f1_score'], result['auc_roc']]
        values += values[:1]

        ax1.plot(angles, values, 'o-', linewidth=2, label=model_name,
                color=colors_radar[idx])
        ax1.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', size=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax1.grid(True)

    ax2 = plt.subplot(2, 3, 2)

    x = np.arange(len(metrics_to_rank))
    width = 0.8 / len(results)
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(results)))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['accuracy'], result['precision'], result['recall'],
                  result['f1_score'], result['auc_roc']]
        offset = (idx - len(results)/2) * width + width/2
        ax2.bar(x + offset, values, width, label=model_name, color=colors_bar[idx])

    ax2.set_xlabel('Metrics', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_to_rank, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')

    ax3 = plt.subplot(2, 3, 3)

    heatmap_data = df_comparison.set_index('Model')[metrics].T
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                cbar_kws={'label': 'Score'}, ax=ax3, vmin=0, vmax=1)
    ax3.set_title('Metrics Heatmap', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Metric', fontsize=12)

    ax4 = plt.subplot(2, 3, 4)

    cm_categories = ['TN', 'FP', 'FN', 'TP']
    x_cm = np.arange(len(cm_categories))

    for idx, (model_name, result) in enumerate(results.items()):
        values = [result['true_negatives'], result['false_positives'],
                  result['false_negatives'], result['true_positives']]
        offset = (idx - len(results)/2) * width + width/2
        ax4.bar(x_cm + offset, values, width, label=model_name, color=colors_bar[idx])

    ax4.set_xlabel('Prediction Type', fontsize=12)
    ax4.set_ylabel('Count', fontsize=12)
    ax4.set_title('Confusion Matrix Components', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_cm)
    ax4.set_xticklabels(cm_categories)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = plt.subplot(2, 3, 5)

    for idx, (model_name, result) in enumerate(results.items()):
        ax5.scatter(result['specificity'], result['sensitivity'],
                   s=200, alpha=0.6, color=colors_bar[idx],
                   label=model_name, edgecolors='black', linewidth=1.5)

    ax5.set_xlabel('Specificity', fontsize=12)
    ax5.set_ylabel('Sensitivity', fontsize=12)
    ax5.set_title('Sensitivity vs Specificity', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([0, 1.05])
    ax5.set_ylim([0, 1.05])
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)

    ax6 = plt.subplot(2, 3, 6)

    df_ranked_plot = df_ranked.sort_values('Overall_Score')
    colors_rank = plt.cm.RdYlGn(df_ranked_plot['Overall_Score'])

    bars = ax6.barh(df_ranked_plot['Model'], df_ranked_plot['Overall_Score'],
                    color=colors_rank, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('Overall Score', fontsize=12)
    ax6.set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
    ax6.set_xlim([0, 1])
    ax6.grid(True, alpha=0.3, axis='x')

    for bar in bars:
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("\n\nComparison visualizations generated!\n")

    best_model = df_ranked.iloc[0]['Model']
    print(f"BEST OVERALL MODEL: {best_model}")
    print(f"Overall Score: {df_ranked.iloc[0]['Overall_Score']:.4f}")
    return df_comparison, best_model
