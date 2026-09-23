import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(group):
    """Calculates evaluation metrics for a specific groupby subset."""
    y_true = group['true_label']
    y_pred = group['pred_label']
    return pd.Series({
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0)
    })

def main():
    predictions_file = 'test_predictions.csv'
    metadata_file = 'AI_Classification-Project.csv'
    
    # Graceful fallback in case original file was named with 'copy'
    if not os.path.exists(metadata_file):
        fallback = 'AI_Classification-Project copy.csv'
        if os.path.exists(fallback):
            metadata_file = fallback
        else:
            print(f"Error: {metadata_file} not found.")
            return

    if not os.path.exists(predictions_file):
        print(f"Error: {predictions_file} not found. Ensure 'train_model.py' ran successfully.")
        return

    print("Loading test predictions and original metadata...")
    preds_df = pd.read_csv(predictions_file)
    meta_df = pd.read_csv(metadata_file)
    
    print("Extracting 'image_id' and merging datasets...")
    # Extract image_id from image_path (e.g., 'dataset/test/REAL/42.jpg' -> 42)
    preds_df['image_id'] = preds_df['image_path'].apply(lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Merge predictions with ground truth demographic attributes
    merged_df = pd.merge(preds_df, meta_df, on='image_id', how='left')
    
    # Target attributes to group by
    attributes = [
        ('gender', 'accuracy_by_gender.png', 'Performance by Gender'),
        ('age_group', 'accuracy_by_age_group.png', 'Performance by Age Group')
    ]
    
    sns.set_theme(style="whitegrid")
    
    for attr, filename, title in attributes:
        if attr not in merged_df.columns:
            print(f"\n[Warning] Column '{attr}' not found in metadata. Skipping {attr}.")
            continue
            
        print(f"\n" + "="*50)
        print(f"{title.upper():^50}")
        print("="*50)
        
        # Calculate all metrics grouped by the specific attribute
        metrics_df = merged_df.groupby(attr).apply(calculate_metrics, include_groups=False).reset_index()
        
        # Print tabular results
        print(metrics_df.to_string(index=False))
        
        # Transform (melt) dataframe for multi-bar seaborn plotting
        melted_df = metrics_df.melt(
            id_vars=attr, 
            value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            var_name='Metric', 
            value_name='Score'
        )
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=melted_df, x=attr, y='Score', hue='Metric', palette='viridis')
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.ylabel('Score (0.0 to 1.0)', fontsize=12)
        plt.xlabel(attr.replace('_', ' ').title(), fontsize=12)
        plt.ylim(0, 1.15) # Leave space for the legend and annotations
        
        # Add exact value annotations atop the bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.2f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=9, fontweight='medium')
                            
        plt.legend(title='Evaluation Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"\n[+] Saved high-res chart to '{filename}'")

if __name__ == "__main__":
    main()
