import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def load_and_analyze_data(file_path):
    """
    Load and perform initial data analysis
    """
    print("Loading data...")
    data = pd.read_csv(file_path)
    
    print("\nInitial Data Analysis:")
    print("-" * 50)
    print(f"Dataset Shape: {data.shape}")
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    # Transaction type distribution
    print("\nTransaction Types Distribution:")
    print(data.type.value_counts())
    
    # Fraud distribution
    print("\nFraud Distribution:")
    print(data.isFraud.value_counts())
    
    return data



def preprocess_data(data, chunk_size=100000):
    """
    Preprocess data with memory optimization
    """
    print("Starting preprocessing...")
    
    # Define datatypes for memory optimization
    dtypes = {
        'type': 'category',
        'amount': 'float32',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'isFraud': 'int8'
    }
    
    # Process in chunks
    chunks = []
    for start_idx in range(0, len(data), chunk_size):
        end_idx = min(start_idx + chunk_size, len(data))
        chunk = data.iloc[start_idx:end_idx].copy()
        
        # Encode transaction types
        type_mapping = {
            "CASH_OUT": 1, 
            "PAYMENT": 2,
            "CASH_IN": 3, 
            "TRANSFER": 4,
            "DEBIT": 5
        }
        
        chunk["type"] = chunk["type"].map(type_mapping)
        
        # Convert datatypes
        for col, dtype in dtypes.items():
            if col in chunk.columns:
                chunk[col] = chunk[col].astype(dtype)
        
        chunks.append(chunk)
        
        # Clear memory
        gc.collect()
        
    processed_data = pd.concat(chunks, axis=0)
    return processed_data


def prepare_features(data, chunk_size=50000):
    """
    Prepare features with memory optimization
    """
    features = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
    
    X_chunks = []
    y_chunks = []
    
    # Initialize transformers
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    for start_idx in range(0, len(data), chunk_size):
        end_idx = min(start_idx + chunk_size, len(data))
        chunk = data.iloc[start_idx:end_idx]
        
        # Extract features
        X_chunk = chunk[features].copy()
        y_chunk = chunk["isFraud"].copy()
        
        # Handle missing values
        X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan)
        X_chunk = pd.DataFrame(
            imputer.fit_transform(X_chunk),
            columns=features,
            index=X_chunk.index
        )
        
        # Scale features
        X_scaled = scaler.fit_transform(X_chunk)
        
        X_chunks.append(X_scaled)
        y_chunks.append(y_chunk.values)
        
        del chunk, X_chunk, y_chunk
        gc.collect()
    
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    return X, y





from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

def train_and_evaluate_models(X, y):
    """
    Comprehensive model training and evaluation with multiple metrics
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Target vector
        
    Returns:
    --------
    tuple : (results, X_test, y_test)
        Dictionary of model results and test data
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        
        model.fit(X_train, y_train)
        
        
        y_pred = model.predict(X_test)
        
        
        training_time = time.time() - start_time
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'training_time': training_time,
            'cv_scores': cv_scores,
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        
        print(f"\n{name} Results:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print("\nClassification Report:")
        print(results[name]['report'])
    
  
    plot_model_comparisons(results, X_test, y_test)
    
    return results, X_test, y_test

def plot_model_comparisons(results, X_test, y_test):
    """
    Create comprehensive visualization of model performance
    """
    
    plt.style.use('seaborn')
    
    # 1. Model Accuracy Comparison
    plt.figure(figsize=(10, 6))
    accuracies = []
    model_names = []
    
    for name, result in results.items():
        report_dict = classification_report(y_test, result['predictions'], output_dict=True)
        accuracies.append(report_dict['accuracy'])
        model_names.append(name)
    
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (name, result) in enumerate(results.items()):
        sns.heatmap(result['confusion_matrix'], 
                   annot=True, 
                   fmt='d', 
                   ax=axes[i],
                   cmap='Blues')
        axes[i].set_title(f'{name} Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # 3. ROC Curves
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        if hasattr(result['model'], "predict_proba"):
            y_prob = result['model'].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    
    # 4. Cross-validation Score Comparison
    plt.figure(figsize=(10, 6))
    cv_means = [result['cv_scores'].mean() for result in results.values()]
    cv_stds = [result['cv_scores'].std() for result in results.values()]
    
    plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
    plt.title('Cross-validation Score Comparison')
    plt.ylabel('Mean CV Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




    if __name__ == "__main__":
    
    CHUNK_SIZE = 100000
    FILE_PATH = "/content/online_payments_fraud_detection_dataset (1).csv"  
    
    
    data = load_and_analyze_data(FILE_PATH)
    
    # Preprocess data
    processed_data = preprocess_data(data, chunk_size=CHUNK_SIZE)
    
    # Prepare features
    X, y = prepare_features(processed_data, chunk_size=CHUNK_SIZE)
    
    # Train and evaluate models
    

    results, X_test, y_test = train_and_evaluate_models(X, y)
    for name, result in results.items():
     print(f"\n{name} Detailed Results:")
     print(f"Training Time: {result['training_time']:.2f} seconds")
     print(f"Mean CV Score: {result['cv_scores'].mean():.3f}")
     print("\nClassification Report:")
     print(result['report'])
    
    print("\nAnalysis Complete!")