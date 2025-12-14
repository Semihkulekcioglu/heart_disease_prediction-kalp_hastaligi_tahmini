"""
Model Yardımcı Fonksiyonları
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os


def train_model(model, X_train, y_train, model_name="Model"):
    """
    Modeli eğitir
    
    Parameters:
    -----------
    model : estimator
        Eğitilecek model
    X_train : array-like
        Eğitim özellikleri
    y_train : array-like
        Eğitim hedef değişkeni
    model_name : str
        Model adı
    
    Returns:
    --------
    model : estimator
        Eğitilmiş model
    """
    print(f"\n{'='*60}")
    print(f"{model_name} EĞİTİLİYOR...")
    print(f"{'='*60}")
    
    model.fit(X_train, y_train)
    
    print(f"✓ {model_name} başarıyla eğitildi!")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Model performansını değerlendirir
    
    Parameters:
    -----------
    model : estimator
        Değerlendirilecek model
    X_train, X_test : array-like
        Eğitim ve test özellikleri
    y_train, y_test : array-like
        Eğitim ve test hedef değişkenleri
    model_name : str
        Model adı
    
    Returns:
    --------
    results : dict
        Performans metrikleri
    """
    # Tahminler
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Olasılık tahminleri (varsa)
    try:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        has_proba = True
    except:
        has_proba = False
    
    # Metrikler
    results = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'test_f1': f1_score(y_test, y_test_pred),
    }
    
    if has_proba:
        results['train_roc_auc'] = roc_auc_score(y_train, y_train_proba)
        results['test_roc_auc'] = roc_auc_score(y_test, y_test_proba)
    
    # Sonuçları yazdır
    print(f"\n{'='*60}")
    print(f"{model_name} PERFORMANS METRİKLERİ")
    print(f"{'='*60}")
    print(f"\n{'Metrik':<20} {'Eğitim':<15} {'Test':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {results['train_accuracy']:.4f}{'':<10} {results['test_accuracy']:.4f}")
    print(f"{'Precision':<20} {results['train_precision']:.4f}{'':<10} {results['test_precision']:.4f}")
    print(f"{'Recall':<20} {results['train_recall']:.4f}{'':<10} {results['test_recall']:.4f}")
    print(f"{'F1-Score':<20} {results['train_f1']:.4f}{'':<10} {results['test_f1']:.4f}")
    if has_proba:
        print(f"{'ROC-AUC':<20} {results['train_roc_auc']:.4f}{'':<10} {results['test_roc_auc']:.4f}")
    print("=" * 60)
    
    return results


def plot_confusion_matrix(y_true, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Confusion matrix'i görselleştirir
    
    Parameters:
    -----------
    y_true : array-like
        Gerçek değerler
    y_pred : array-like
        Tahmin edilen değerler
    model_name : str
        Model adı
    figsize : tuple
        Grafik boyutu
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Sağlıklı', 'Hasta'], 
                yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Gerçek Değer', fontsize=12)
    plt.xlabel('Tahmin', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Detaylı bilgi
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Detayları:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")


def plot_roc_curve(y_true, y_proba, model_name="Model", figsize=(8, 6)):
    """
    ROC eğrisini çizer
    
    Parameters:
    -----------
    y_true : array-like
        Gerçek değerler
    y_proba : array-like
        Pozitif sınıf olasılıkları
    model_name : str
        Model adı
    figsize : tuple
        Grafik boyutu
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5, scoring='accuracy', model_name="Model"):
    """
    Cross-validation ile model performansını değerlendirir
    
    Parameters:
    -----------
    model : estimator
        Değerlendirilecek model
    X : array-like
        Özellikler
    y : array-like
        Hedef değişken
    cv : int
        Fold sayısı
    scoring : str
        Değerlendirme metriği
    model_name : str
        Model adı
    
    Returns:
    --------
    scores : array
        CV skorları
    """
    print(f"\n{model_name} - {cv}-Fold Cross Validation yapılıyor...")
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"\nCross Validation Sonuçları:")
    print(f"  Skorlar: {scores}")
    print(f"  Ortalama: {scores.mean():.4f}")
    print(f"  Standart Sapma: {scores.std():.4f}")
    
    return scores


def tune_hyperparameters(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', model_name="Model"):
    """
    GridSearchCV ile hiperparametre optimizasyonu yapar
    
    Parameters:
    -----------
    model : estimator
        Optimize edilecek model
    param_grid : dict
        Parametre grid'i
    X_train : array-like
        Eğitim özellikleri
    y_train : array-like
        Eğitim hedef değişkeni
    cv : int
        Fold sayısı
    scoring : str
        Değerlendirme metriği
    model_name : str
        Model adı
    
    Returns:
    --------
    best_model : estimator
        En iyi model
    best_params : dict
        En iyi parametreler
    """
    print(f"\n{'='*60}")
    print(f"{model_name} - HİPERPARAMETRE OPTİMİZASYONU")
    print(f"{'='*60}")
    print(f"Toplam {len(param_grid)} parametre test edilecek...")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, 
        n_jobs=-1, verbose=1, return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Optimizasyon tamamlandı!")
    print(f"\nEn İyi Parametreler:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\nEn İyi {scoring.capitalize()} Skoru: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def save_model(model, filename, folder='../models'):
    """
    Modeli kaydeder
    
    Parameters:
    -----------
    model : estimator
        Kaydedilecek model
    filename : str
        Dosya adı
    folder : str
        Klasör yolu
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    joblib.dump(model, filepath)
    print(f"✓ Model kaydedildi: {filepath}")


def load_model(filename, folder='../models'):
    """
    Kaydedilmiş modeli yükler
    
    Parameters:
    -----------
    filename : str
        Dosya adı
    folder : str
        Klasör yolu
    
    Returns:
    --------
    model : estimator
        Yüklenen model
    """
    filepath = os.path.join(folder, filename)
    model = joblib.load(filepath)
    print(f"✓ Model yüklendi: {filepath}")
    return model


def plot_feature_importance(model, feature_names, top_n=None, figsize=(10, 6)):
    """
    Feature importance'ı görselleştirir
    
    Parameters:
    -----------
    model : estimator
        Feature importance'a sahip model
    feature_names : list
        Özellik isimleri
    top_n : int, optional
        Gösterilecek en önemli n özellik
    figsize : tuple
        Grafik boyutu
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("⚠ Bu model feature importance desteklemiyor.")
        return
    
    # DataFrame oluştur
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    if top_n:
        feature_importance_df = feature_importance_df.head(top_n)
        title = f'Top {top_n} Feature Importance'
    else:
        title = 'Feature Importance'
    
    # Görselleştirme
    plt.figure(figsize=figsize)
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df


def compare_models(results_dict, metric='test_accuracy', figsize=(12, 6)):
    """
    Farklı modelleri karşılaştırır
    
    Parameters:
    -----------
    results_dict : dict
        Model adı: sonuçlar sözlüğü
    metric : str
        Karşılaştırılacak metrik
    figsize : tuple
        Grafik boyutu
    """
    models = list(results_dict.keys())
    scores = [results_dict[model][metric] for model in models]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # En iyi modeli vurgula
    best_idx = np.argmax(scores)
    bars[best_idx].set_color('orange')
    
    plt.title(f'Model Karşılaştırması - {metric.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([min(scores) - 0.05, max(scores) + 0.05])
    
    # Değerleri yazdır
    for i, (model, score) in enumerate(zip(models, scores)):
        plt.text(i, score + 0.01, f'{score:.4f}', ha='center', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # En iyi modeli belirt
    best_model = models[best_idx]
    print(f"\n🏆 En iyi model: {best_model} ({metric}: {scores[best_idx]:.4f})")


