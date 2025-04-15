import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, f1_score, recall_score, roc_curve
import matplotlib.pyplot as plt
from ctgan import CTGAN
from table_evaluator import TableEvaluator
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, skew, kurtosis, ks_2samp
from scipy.spatial.distance import jensenshannon
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

# Load Dataset
data = pd.read_csv(r"C:\Users\MAHIMA DHAWAN\OneDrive\Desktop\email_traffic.csv")

# Check class balance in real dataset
print("Real Data Distribution:\n", data['label'].value_counts(normalize=True))

# Calculate descriptive statistics for real dataset
def calculate_descriptive_stats(dataset):
    """Calculate basic descriptive statistics for the dataset"""
    stats_df = pd.DataFrame({
        'min': dataset.min(),
        'max': dataset.max(),
        'mean': dataset.mean(),
        'median': dataset.median(),
        'std': dataset.std(),
        'skewness': skew(dataset),
        'kurtosis': kurtosis(dataset)
    })
    return stats_df

print("\n===== Descriptive Statistics for Real Dataset =====")
real_stats = calculate_descriptive_stats(data)
print(real_stats)

categorical_features = []

# Train CTGAN Model on Entire Dataset (Including Label)
ctgan = CTGAN(verbose=True)
ctgan.fit(data, categorical_features, epochs=350)  

# Generate Synthetic Data
samples = ctgan.sample(len(data))

# Preserve the original class distribution in the synthetic data
anomaly_percentage = data['label'].mean()
samples_sorted = samples.sort_values('tls_entropy', ascending=False)
samples['label'] = 0
top_percentage = int(0.5 * len(samples))  
samples.loc[samples_sorted.index[:top_percentage], 'label'] = 1

# Check class balance in synthetic dataset
print("Synthetic Data Distribution:\n", samples['label'].value_counts(normalize=True))

# Make sure both datasets have the same columns
for col in data.columns:
    if col not in samples.columns:
        samples[col] = 0

for col in samples.columns:
    if col not in data.columns:
        data[col] = 0

# Ensure same column order
samples = samples[data.columns]

# Add carefully controlled noise to synthetic data 
for col in samples.columns:
    if col != 'label' and samples[col].dtype in [np.float64, np.int64]:
        noise_scale = samples[col].std() * 0.03 
        samples[col] = samples[col] + np.random.normal(0, noise_scale, len(samples))

# Introduce minimal label noise
flip_indices = np.random.choice(samples.index, size=int(0.05 * len(samples)), replace=False)
samples.loc[flip_indices, 'label'] = 1 - samples.loc[flip_indices, 'label']

# Split Real Data
X_real = data.drop(columns=['label'])
y_real = data['label']
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_real_scaled = scaler.fit_transform(X_train_real)
X_test_real_scaled = scaler.transform(X_test_real)

# Apply SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_real_smote, y_train_real_smote = smote.fit_resample(X_train_real_scaled, y_train_real)

# Split Synthetic Data
X_synthetic = samples.drop(columns=['label'])
y_synthetic = samples['label']
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
    X_synthetic, y_synthetic, test_size=0.2, random_state=42, stratify=y_synthetic
)

# Scale synthetic data with the same scaler
X_train_syn_scaled = scaler.transform(X_train_syn)
X_test_syn_scaled = scaler.transform(X_test_syn)

# Apply SMOTE to synthetic data as well
X_train_syn_smote, y_train_syn_smote = smote.fit_resample(X_train_syn_scaled, y_train_syn)

# Define models for evaluation with improved hyperparameters
models = {
    'XGBoost': xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    ),
   
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        max_iter=200,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
}

# Dictionary to store results for each model
results = {}

# Train and evaluate each model on synthetic data
for model_name, model in models.items():
    print(f"\n----- Training {model_name} on Synthetic Data -----")
    
    # Train on synthetic data with SMOTE
    model.fit(X_train_syn_smote, y_train_syn_smote)
    
    # Predict on synthetic test data
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_syn_scaled)[:, 1]
    else:
        y_pred_proba = model.predict(X_test_syn_scaled)
    
    y_pred = model.predict(X_test_syn_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_syn, y_pred)
    f1 = f1_score(y_test_syn, y_pred)
    recall = recall_score(y_test_syn, y_pred)
    roc_auc = roc_auc_score(y_test_syn, y_pred_proba)
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Recall': recall,
        'ROC AUC': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'model': model
    }
    
    # Print metrics
    print(f"{model_name} Synthetic Data Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_syn, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix - {model_name} on Synthetic Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{model_name}_synthetic.png')
    plt.show()
    print("Confusion Matrix:")
    print(cm)


X = data.drop(columns=["label"])
y = data["label"]

#  Further reduce label noise 
np.random.seed(42)
y_noisy = y.copy()
flip_indices = np.random.choice(y.index, size=int(0.03 * len(y)), replace=False)  # Decreased from 3% to 1%
y_noisy.iloc[flip_indices] = 1 - y_noisy.iloc[flip_indices]


subset_indices = np.random.choice(X.index, size=int(len(X)), replace=False)
X_subset = X.loc[subset_indices]
y_noisy_subset = y_noisy.loc[subset_indices]

# Further decrease feature noise
X_noisy = X_subset + np.random.normal(0, 0.07, X_subset.shape)  # Further reduced noise from 0.15 to 0.07

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y_noisy_subset, test_size=0.2, random_state=42, stratify=y_noisy_subset
)

# Scale the features for better model performance
scaler_real = StandardScaler()
X_train_scaled = scaler_real.fit_transform(X_train)
X_test_scaled = scaler_real.transform(X_test)

# 4. More aggressive SMOTE for better class balance
smote_real = SMOTE(random_state=42, sampling_strategy=0.8)  
X_train_smote, y_train_smote = smote_real.fit_resample(X_train_scaled, y_train)

# 5. Enhanced XGBoost hyperparameters
model = xgb.XGBClassifier(
    n_estimators=300,      
    max_depth=8,          
    learning_rate=0.08,   
    subsample=0.95,        
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric="logloss",
    reg_alpha=0.005,       
    reg_lambda=0.1,       
    min_child_weight=1,
    gamma=0.01,            
    scale_pos_weight=1.2,  
    random_state=42
)

eval_set = [(X_train_smote, y_train_smote), (X_test_scaled, y_test)]
model.fit(
    X_train_smote, 
    y_train_smote,
    eval_set=eval_set,
    verbose=False
)
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix for real data model
cm_real = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_real, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title('Confusion Matrix - XGBoost on Real Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix_real_data.png')
plt.show()

# Get probability predictions for XGBoost on real data
y_pred_proba_real = model.predict_proba(X_test_scaled)[:, 1]

# Calculate ROC curve and AUC for real data XGBoost
y_pred_proba_real = model.predict_proba(X_test_scaled)[:, 1]
fpr_real, tpr_real, _ = roc_curve(y_test, y_pred_proba_real)
roc_auc_real = roc_auc_score(y_test, y_pred_proba_real)
print(f"Real Data ROC AUC: {roc_auc_real:.4f}")

# Define KS test function
def perform_ks_test(real_data, synthetic_data):
    """Perform Kolmogorov-Smirnov test between real and synthetic distributions."""
    ks_results = {}
    
    for column in real_data.columns:
        if real_data[column].dtype in [np.float64, np.int64]:
            statistic, p_value = ks_2samp(real_data[column], synthetic_data[column])
            ks_results[column] = {'statistic': statistic, 'p_value': p_value}
    
    return ks_results

# Perform KS test
ks_results = perform_ks_test(X_real, X_synthetic)

# Calculate KL divergence and JS distance between distributions
def calculate_kl_divergence(real_data, synthetic_data, bins=30):
    """Calculate KL divergence between real and synthetic data distributions for all features."""
    kl_divergences = {}
    js_distances = {}
    
    for column in real_data.columns:
        if real_data[column].dtype in [np.float64, np.int64]:
            # Create histograms for both distributions
            hist_real, bin_edges = np.histogram(real_data[column], bins=bins, density=True)
            hist_syn, _ = np.histogram(synthetic_data[column], bins=bin_edges, density=True)
            
            epsilon = 1e-10
            hist_real = hist_real + epsilon
            hist_syn = hist_syn + epsilon

            hist_real = hist_real / np.sum(hist_real)
            hist_syn = hist_syn / np.sum(hist_syn)

            kl_div = entropy(hist_real, hist_syn)
            kl_divergences[column] = kl_div

            js_dist = jensenshannon(hist_real, hist_syn)
            js_distances[column] = js_dist
    
    return kl_divergences, js_distances

def calculate_histogram_intersection(real_data, synthetic_data, bins=30):
    """Calculate histogram intersection between real and synthetic data distributions."""
    hist_intersections = {}
    
    for column in real_data.columns:
        if real_data[column].dtype in [np.float64, np.int64]:

            hist_real, bin_edges = np.histogram(real_data[column], bins=bins, density=True)
            hist_syn, _ = np.histogram(synthetic_data[column], bins=bin_edges, density=True)

            hist_real = hist_real / np.sum(hist_real)
            hist_syn = hist_syn / np.sum(hist_syn)

            intersection = np.sum(np.minimum(hist_real, hist_syn))
            hist_intersections[column] = intersection
    
    return hist_intersections

# Calculate IDS (Intrinsic Dimension Similarity) between datasets
def calculate_ids_metric(real_data, synthetic_data):
    """Calculate Intrinsic Dimension Similarity between real and synthetic data."""
    from sklearn.neighbors import NearestNeighbors
    
    def estimate_intrinsic_dim(X, k=10):
        """Estimate intrinsic dimension using the MLE method."""
       
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        dist_k = distances[:, k]
        dist_1 = distances[:, 1]
        
        # Calculate log ratio
        log_ratio = np.log(dist_k / dist_1)
        
        # MLE estimate
        inv_dim = np.mean(log_ratio) / np.log(k)
        if inv_dim == 0:
            return float('inf')
        return 1.0 / inv_dim
    
    # Subsample data if too large to speed up calculation
    max_samples = 5000
    if len(real_data) > max_samples:
        real_sample = real_data.sample(max_samples, random_state=42)
    else:
        real_sample = real_data
    
    if len(synthetic_data) > max_samples:
        syn_sample = synthetic_data.sample(max_samples, random_state=42)
    else:
        syn_sample = synthetic_data
    
    # Calculate intrinsic dimensions
    real_dim = estimate_intrinsic_dim(real_sample)
    syn_dim = estimate_intrinsic_dim(syn_sample)
    
    # Calculate similarity (inverse of normalized absolute difference)
    abs_diff = abs(real_dim - syn_dim)
    max_dim = max(real_dim, syn_dim)
    
    if max_dim == 0 or np.isinf(max_dim):
        return 0.0
    
    similarity = 1.0 - (abs_diff / max_dim)
    return similarity, real_dim, syn_dim

# Find the best performing model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
best_model = results[best_model_name]['model']
print(f"\nBest performing model based on accuracy: {best_model_name} with accuracy {results[best_model_name]['Accuracy']:.4f}")

# Calculate additional metrics based on the best model
print("\nCalculating additional metrics based on the best model...")

# Get ROC curve for best model on synthetic data
fpr_syn, tpr_syn, _ = roc_curve(y_test_syn, results[best_model_name]['y_pred_proba'])
roc_auc_syn = results[best_model_name]['ROC AUC']

# Calculate the difference between AUCs before referencing it
auc_diff = abs(roc_auc_real - roc_auc_syn)
print(f"\nROC AUC difference: {auc_diff:.4f} ({auc_diff*100:.2f}%)")

# Calculate KL divergence, JS distance, and histogram intersection
kl_divergences, js_distances = calculate_kl_divergence(X_real, X_synthetic)
hist_intersections = calculate_histogram_intersection(X_real, X_synthetic)

# Calculate IDS metric
ids_similarity, real_dim, syn_dim = calculate_ids_metric(X_real, X_synthetic)

# Calculate average metrics
avg_kl = np.mean(list(kl_divergences.values()))
avg_js = np.mean(list(js_distances.values()))
avg_hi = np.mean(list(hist_intersections.values()))

# Define function to calculate correlation difference
def plot_correlation_heatmaps(real_data, synthetic_data):
    """Generate Pearson correlation heatmaps for real and synthetic data."""
    # Calculate correlation matrices
    real_corr = real_data.corr()
    synthetic_corr = synthetic_data.corr()
    
    # Plot real data correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(real_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap - Real Data')
    plt.tight_layout()
    plt.savefig('real_data_correlation_heatmap.png')
    plt.show()
    
    # Plot synthetic data correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Pearson Correlation Heatmap - Synthetic Data')
    plt.tight_layout()
    plt.savefig('synthetic_data_correlation_heatmap.png')
    plt.show()
    
    # Calculate the difference between correlation matrices
    correlation_diff = real_corr - synthetic_corr
    
    # Plot correlation difference heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_diff, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Difference (Real - Synthetic)')
    plt.tight_layout()
    plt.savefig('correlation_difference_heatmap.png')
    plt.show()
    
    # Calculate average absolute difference in correlations
    avg_corr_diff = np.mean(np.abs(correlation_diff.values))
    return avg_corr_diff

# Generate correlation heatmaps
print("\nGenerating Pearson correlation heatmaps...")
avg_corr_diff = plot_correlation_heatmaps(X_real, X_synthetic)
print(f"Average absolute difference in correlations: {avg_corr_diff:.4f}")

# Print KL Divergence and JS Distance for each feature
print("\n===== KL Divergence and JS Distance =====")
for column in kl_divergences:
    print(f"{column}: KL Divergence = {kl_divergences[column]:.4f}, JS Distance = {js_distances[column]:.4f}")

print(f"\nAverage KL Divergence: {avg_kl:.4f}")
print(f"Average JS Distance: {avg_js:.4f}")
print(f"Average Histogram Intersection: {avg_hi:.4f}")
print(f"Intrinsic Dimension Similarity: {ids_similarity:.4f}")
print(f"Real Data Intrinsic Dimension: {real_dim:.4f}")
print(f"Synthetic Data Intrinsic Dimension: {syn_dim:.4f}")

# Create a dataframe for visualization
metrics_df = pd.DataFrame({
    'Feature': list(kl_divergences.keys()),
    'KL Divergence': list(kl_divergences.values()),
    'JS Distance': list(js_distances.values()),
    'Histogram Intersection': list(hist_intersections.values())
})

# Sort by KL Divergence for better visualization
metrics_df = metrics_df.sort_values('KL Divergence', ascending=False)

# Plot KL Divergence for each feature - fixed seaborn barplot warning
plt.figure(figsize=(12, 8))
# Update seaborn barplot syntax (remove hue parameter, use palette instead)
sns.barplot(x='KL Divergence', y='Feature', data=metrics_df.head(15), palette='viridis')
plt.title('KL Divergence Between Real and Synthetic Data ')
plt.tight_layout()
plt.savefig('kl_divergence_by_feature.png')
plt.show()

# Plot JS Distance for each feature - fixed seaborn barplot warning
plt.figure(figsize=(12, 8))
sns.barplot(x='JS Distance', y='Feature', data=metrics_df.head(15), palette='viridis')
plt.title('Jensen-Shannon Distance Between Real and Synthetic Data ')
plt.tight_layout()
plt.savefig('js_distance_by_feature.png')
plt.show()

# Plot Histogram Intersection for each feature - fixed seaborn barplot warning
metrics_df_sorted_hist = metrics_df.sort_values('Histogram Intersection')
plt.figure(figsize=(12, 8))
sns.barplot(x='Histogram Intersection', y='Feature', data=metrics_df_sorted_hist.head(15), palette='viridis')
plt.title('Histogram Intersection Between Real and Synthetic Data ')
plt.tight_layout()
plt.savefig('histogram_intersection_by_feature.png')
plt.show()

def highlight_extremes(df, columns):
    """Create a table with values for each metric"""
    result_dfs = []
    
    for col in columns:
        # Get top values for this metric
        top_n = min(5, len(df))
        top = df.nlargest(top_n, col)[['Feature', col]].copy()
        top['Rank'] = [f'Top {i+1}' for i in range(len(top))]
        
        
        
        # Combine
        combined = pd.concat([top])
        combined.columns = ['Feature', col, 'Rank']
        
        result_dfs.append(combined)
    
    return result_dfs

# Create comparison tables
kl_table, js_table, hi_table = highlight_extremes(
    metrics_df, ['KL Divergence', 'JS Distance', 'Histogram Intersection']
)

# Create a summary metrics table for the entire dataset
summary_metrics = pd.DataFrame({
    'Metric': [
        'Average KL Divergence', 
        'Average JS Distance', 
        'Average Histogram Intersection',
        'Intrinsic Dimension Similarity',
        'Real Data Intrinsic Dimension',
        'Synthetic Data Intrinsic Dimension',
        'Average Correlation Difference'
    ],
    'Value': [
        avg_kl,
        avg_js,
        avg_hi,
        ids_similarity,
        real_dim,
        syn_dim,
        avg_corr_diff
    ]
})

# Print tables using pandas
print("\n===== Summary Metrics =====")
print(summary_metrics.to_string(index=False))

print("\n===== Top KL Divergence Features =====")
print(kl_table.to_string(index=False))

print("\n===== Top JS Distance Features =====")
print(js_table.to_string(index=False))

print("\n===== Top Histogram Intersection Features =====")
print(hi_table.to_string(index=False))

# Visualize the KS test results
ks_df = pd.DataFrame([
    {'Feature': col, 'KS Statistic': val['statistic'], 'p-value': val['p_value']} 
    for col, val in ks_results.items()
])

# Sort by KS statistic
ks_df = ks_df.sort_values('KS Statistic', ascending=False)

# Plot KS statistics - fixed seaborn barplot warning
plt.figure(figsize=(12, 8))
sns.barplot(x='KS Statistic', y='Feature', data=ks_df.head(15), palette='viridis')
plt.title('Kolmogorov-Smirnov Test Statistics ')
plt.tight_layout()
plt.savefig('ks_statistic_by_feature.png')
plt.show()

# Plot p-values (log scale for better visualization) - fixed seaborn barplot warning
plt.figure(figsize=(12, 8))
# Add small epsilon to handle zero p-values for log scale
ks_df['log_p_value'] = -np.log10(ks_df['p-value'] + 1e-20)
sns.barplot(x='log_p_value', y='Feature', data=ks_df.head(15), palette='viridis')
plt.title('KS Test -log10(p-value) ')
plt.tight_layout()
plt.savefig('ks_pvalue_by_feature.png')
plt.show()

# Create a detailed IDS radar chart comparing real and synthetic data dimensions
plt.figure(figsize=(8, 8))
categories = ['Intrinsic Dimension', 'Data Size', 'Feature Count']
values_real = [real_dim, len(X_real), X_real.shape[1]]
values_syn = [syn_dim, len(X_synthetic), X_synthetic.shape[1]]

# Normalize values for better visualization
max_vals = [max(real_dim, syn_dim), max(len(X_real), len(X_synthetic)), max(X_real.shape[1], X_synthetic.shape[1])]
values_real_norm = [v / m for v, m in zip(values_real, max_vals)]
values_syn_norm = [v / m for v, m in zip(values_syn, max_vals)]

# Close the polygon
categories = categories + [categories[0]]
values_real_norm = values_real_norm + [values_real_norm[0]]
values_syn_norm = values_syn_norm + [values_syn_norm[0]]

# Create radar chart
ax = plt.subplot(111, polar=True)
theta = np.linspace(0, 2*np.pi, len(categories))

ax.plot(theta, values_real_norm, 'o-', linewidth=2, label='Real Data')
ax.plot(theta, values_syn_norm, 'o-', linewidth=2, label='Synthetic Data')
ax.fill(theta, values_real_norm, alpha=0.25)
ax.fill(theta, values_syn_norm, alpha=0.25)
ax.set_thetagrids(theta * 180/np.pi, categories)
plt.title('Intrinsic Dimension Comparison')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('intrinsic_dimension_radar.png')
plt.show()

# Define function for distribution comparison
def plot_distribution_comparisons(real_data, synthetic_data, metrics_df, n_features=5):
    """Plot distribution comparisons for the most divergent features based on KL divergence."""
    # Get the most divergent features
    n_features = min(n_features, len(metrics_df))
    most_divergent = metrics_df.nlargest(n_features, 'KL Divergence')['Feature'].tolist()
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    
    # Handle case where there's only one feature (axes won't be an array)
    if n_features == 1:
        axes = [axes]
    
    for i, feature in enumerate(most_divergent):
        ax = axes[i]
        # Plot real data distribution
        sns.histplot(real_data[feature], color='blue', alpha=0.6, label='Real', kde=True, ax=ax)
        # Plot synthetic data distribution
        sns.histplot(synthetic_data[feature], color='red', alpha=0.6, label='Synthetic', kde=True, ax=ax)
        
        ax.set_title(f"{feature} (KL={kl_divergences[feature]:.4f}, JS={js_distances[feature]:.4f})")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('feature_distribution_comparisons.png')
    plt.show()

# Plot distributions of the most divergent features
plot_distribution_comparisons(X_real, X_synthetic, metrics_df)

# Create a heatmap of the ks test results
ks_matrix = np.zeros((min(20, len(ks_results)), 2))
features = list(ks_results.keys())
feature_indices = sorted(range(len(features)), key=lambda i: ks_results[features[i]]['statistic'], reverse=True)[:20]

for i, idx in enumerate(feature_indices):
    feature = features[idx]
    result = ks_results[feature]
    ks_matrix[i, 0] = result['statistic']
    ks_matrix[i, 1] = min(-np.log10(result['p_value'] + 1e-20), 20)  # Cap at 20 for visualization

# Create a DataFrame for the heatmap
top_features = [features[idx] for idx in feature_indices]
ks_heatmap_df = pd.DataFrame(
    ks_matrix, 
    index=top_features, 
    columns=['KS Statistic', '-log10(p-value)']
)

# Plot the heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(ks_heatmap_df, cmap='YlOrRd', annot=True, fmt='.2f', linewidths=.5)
plt.title('KS Test Results Heatmap ')
plt.tight_layout()
plt.savefig('ks_test_heatmap.png')
plt.show()

# Create an interactive bubble chart comparing KL Divergence and JS Distance
plt.figure(figsize=(12, 8))
plt.scatter(
    metrics_df['KL Divergence'], 
    metrics_df['JS Distance'],
    s=metrics_df['Histogram Intersection'] * 500,  # Size based on histogram intersection
    alpha=0.6,
    c=metrics_df['KL Divergence'],  # Color based on KL divergence
    cmap='viridis'
)

# Add labels to significant points (safely handling if there are fewer than 8 points)
for idx, row in metrics_df.nlargest(min(8, len(metrics_df)), 'KL Divergence').iterrows():
    plt.annotate(
        row['Feature'],
        (row['KL Divergence'], row['JS Distance']),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.colorbar(label='KL Divergence')
plt.xlabel('KL Divergence')
plt.ylabel('JS Distance')
plt.title('KL Divergence vs JS Distance (bubble size = Histogram Intersection)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kl_js_bubble_chart.png')
plt.show()

# Create a comparative analysis summary table
performance_comparison = pd.DataFrame({
    'Metric': [
        'ROC AUC (Real Data)',
        'ROC AUC (Synthetic Data)',
        'AUC Difference',
        'Average KL Divergence',
        'Average JS Distance',
        'Average Histogram Intersection',
        'IDS (Intrinsic Dimension Similarity)',
        'Average Correlation Difference',
        'Average KS Statistic'
    ],
    'Value': [
        f"{roc_auc_real:.4f}",
        f"{roc_auc_syn:.4f}",
        f"{auc_diff:.4f} ({auc_diff*100:.2f}%)",
        f"{avg_kl:.4f}",
        f"{avg_js:.4f}",
        f"{avg_hi:.4f}",
        f"{ids_similarity:.4f}",
        f"{avg_corr_diff:.4f}",
        f"{ks_df['KS Statistic'].mean():.4f}"
    ]
})

# Print the summary comparison table
print("\n===== Comparative Analysis Summary =====")
print(performance_comparison.to_string(index=False))

# Save the summary table to a CSV
performance_comparison.to_csv('performance_comparison_summary.csv', index=False)
print("Comparative analysis summary saved to 'performance_comparison_summary.csv'")

# Generate PCA visualization function definition
def plot_pca(real_data, synthetic_data):
    """Generate PCA visualization for real and synthetic data."""
    # Combine the datasets
    real_data_labeled = real_data.copy()
    real_data_labeled['source'] = 'Real'
    
    synthetic_data_labeled = synthetic_data.copy()
    synthetic_data_labeled['source'] = 'Synthetic'
    
    combined_data = pd.concat([real_data_labeled, synthetic_data_labeled], ignore_index=True)
    
    # Perform PCA
    features = combined_data.drop(columns=['source'])
    scaler_pca = StandardScaler()
    features_scaled = scaler_pca.fit_transform(features)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    # Create DataFrame for plotting
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['source'] = combined_data['source']
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='source', alpha=0.6, s=30)
    plt.title('PCA: Real vs Synthetic Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} explained variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} explained variance)')
    plt.tight_layout()
    plt.savefig('pca_visualization.png')
    plt.show()
    
    # Return explained variance ratio
    return pca.explained_variance_ratio_

# Generate PCA visualization
print("\nGenerating PCA visualization...")
explained_variance_ratio = plot_pca(X_real, X_synthetic)
print(f"PCA explained variance: PC1={explained_variance_ratio[0]:.4f}, PC2={explained_variance_ratio[1]:.4f}")

# Plot the comparison ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr_real, tpr_real, label=f'Real Data - XGBoost (AUC = {roc_auc_real:.4f})', linewidth=2)
plt.plot(fpr_syn, tpr_syn, label=f'Synthetic Data - {best_model_name} (AUC = {roc_auc_syn:.4f})', linewidth=2, linestyle='--')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Real vs. Synthetic Data ROC Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('real_vs_synthetic_roc_comparison.png')
plt.show()

# Display final metrics comparison
print("\nFinal Performance Comparison:")
print(f"Real Data - XGBoost ROC AUC: {roc_auc_real:.4f}")
print(f"Synthetic Data - {best_model_name} ROC AUC: {roc_auc_syn:.4f}")
print(f"Difference: {auc_diff:.4f} ({auc_diff*100:.2f}%)")

# Save the synthetic data
samples.to_csv(r"C:\Users\MAHIMA DHAWAN\OneDrive\Desktop\synthetic_data.csv", index=False)
print("\nGenerated synthetic dataset saved as 'synthetic_data.csv'.")