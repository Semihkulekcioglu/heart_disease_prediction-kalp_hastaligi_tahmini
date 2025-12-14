"""
Kalan model notebook'larını otomatik oluşturan script
"""

import nbformat as nbf

# Notebook şablonları
models_config = [
    {
        'filename': 'notebooks/03_knn.ipynb',
        'title': 'k-Nearest Neighbors (kNN)',
        'model_import': 'from sklearn.neighbors import KNeighborsClassifier',
        'model_creation': 'knn_model = KNeighborsClassifier(n_neighbors=5)',
        'model_name': 'k-Nearest Neighbors',
        'model_var': 'knn_model',
        'param_grid': """param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}""",
        'save_name': 'knn_model.pkl'
    },
    {
        'filename': 'notebooks/04_decision_tree.ipynb',
        'title': 'Decision Tree',
        'model_import': 'from sklearn.tree import DecisionTreeClassifier',
        'model_creation': 'dt_model = DecisionTreeClassifier(random_state=42)',
        'model_name': 'Decision Tree',
        'model_var': 'dt_model',
        'param_grid': """param_grid = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}""",
        'save_name': 'decision_tree_model.pkl'
    },
    {
        'filename': 'notebooks/05_random_forest.ipynb',
        'title': 'Random Forest',
        'model_import': 'from sklearn.ensemble import RandomForestClassifier',
        'model_creation': 'rf_model = RandomForestClassifier(random_state=42, n_estimators=100)',
        'model_name': 'Random Forest',
        'model_var': 'rf_model',
        'param_grid': """param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}""",
        'save_name': 'random_forest_model.pkl',
        'has_feature_importance': True
    },
    {
        'filename': 'notebooks/06_lightgbm.ipynb',
        'title': 'LightGBM',
        'model_import': 'import lightgbm as lgb',
        'model_creation': 'lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)',
        'model_name': 'LightGBM',
        'model_var': 'lgb_model',
        'param_grid': """param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}""",
        'save_name': 'lightgbm_model.pkl',
        'has_feature_importance': True
    },
    {
        'filename': 'notebooks/07_xgboost.ipynb',
        'title': 'XGBoost',
        'model_import': 'import xgboost as xgb',
        'model_creation': 'xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="logloss")',
        'model_name': 'XGBoost',
        'model_var': 'xgb_model',
        'param_grid': """param_grid = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 0.9, 1.0]
}""",
        'save_name': 'xgboost_model.pkl',
        'has_feature_importance': True
    }
]


def create_model_notebook(config):
    """Model notebook'u oluşturur"""
    nb = nbf.v4.new_notebook()
    cells = []
    
    # Başlık
    cells.append(nbf.v4.new_markdown_cell(
        f"# 🫀 {config['title']} - Kalp Hastalığı Tahmini\n\n"
        f"Bu notebook'ta **{config['title']}** algoritmasını kullanarak kalp hastalığı tahmini yapacağız."
    ))
    
    # Kütüphaneler
    cells.append(nbf.v4.new_code_cell(
        "# Kütüphaneler\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "# Model\n"
        f"{config['model_import']}\n\n"
        "# Kendi modüllerimiz\n"
        "import sys\n"
        "sys.path.append('../src')\n"
        "from preprocessing import load_data, split_data, scale_features\n"
        "from model_utils import (train_model, evaluate_model, plot_confusion_matrix, \n"
        "                         plot_roc_curve, cross_validate_model, tune_hyperparameters,\n"
        "                         save_model"
        + (", plot_feature_importance)\n\n" if config.get('has_feature_importance') else ")\n\n")
        + "print('✓ Tüm kütüphaneler yüklendi!')"
    ))
    
    # Veri yükleme
    cells.append(nbf.v4.new_markdown_cell("## 1. Veri Yükleme ve Hazırlık"))
    cells.append(nbf.v4.new_code_cell(
        "# Veri yükleme\n"
        "df = load_data('../data/heart_disease.csv')\n\n"
        "# Train-test split\n"
        "X_train, X_test, y_train, y_test = split_data(df, target_column='target', test_size=0.2, random_state=42)\n\n"
        "# Feature scaling\n"
        "X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')"
    ))
    
    # Model eğitimi
    cells.append(nbf.v4.new_markdown_cell("## 2. Model Eğitimi"))
    cells.append(nbf.v4.new_code_cell(
        f"# {config['title']} modeli oluştur\n"
        f"{config['model_creation']}\n\n"
        f"# Modeli eğit\n"
        f"{config['model_var']} = train_model({config['model_var']}, X_train_scaled, y_train, model_name='{config['model_name']}')"
    ))
    
    # Değerlendirme
    cells.append(nbf.v4.new_markdown_cell("## 3. Model Değerlendirmesi"))
    cells.append(nbf.v4.new_code_cell(
        f"# Model performansını değerlendir\n"
        f"results = evaluate_model({config['model_var']}, X_train_scaled, X_test_scaled, y_train, y_test, \n"
        f"                        model_name='{config['model_name']}')"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        f"# Confusion Matrix\n"
        f"y_pred = {config['model_var']}.predict(X_test_scaled)\n"
        f"plot_confusion_matrix(y_test, y_pred, model_name='{config['model_name']}')"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        f"# ROC Curve\n"
        f"y_proba = {config['model_var']}.predict_proba(X_test_scaled)[:, 1]\n"
        f"plot_roc_curve(y_test, y_proba, model_name='{config['model_name']}')"
    ))
    
    # Feature importance (eğer varsa)
    if config.get('has_feature_importance'):
        cells.append(nbf.v4.new_markdown_cell("## 4. Feature Importance"))
        cells.append(nbf.v4.new_code_cell(
            f"# Feature importance görselleştirmesi\n"
            f"feature_names = X_train.columns.tolist()\n"
            f"plot_feature_importance({config['model_var']}, feature_names, top_n=10)"
        ))
        section_num = 5
    else:
        section_num = 4
    
    # Hiperparametre optimizasyonu
    cells.append(nbf.v4.new_markdown_cell(f"## {section_num}. Hiperparametre Optimizasyonu"))
    cells.append(nbf.v4.new_code_cell(
        f"# Parametre grid\n"
        f"{config['param_grid']}\n\n"
        f"# GridSearchCV\n"
        f"best_model, best_params = tune_hyperparameters(\n"
        f"    {config['model_creation'].split('=')[1].strip()},\n"
        f"    param_grid, X_train_scaled, y_train, cv=5, \n"
        f"    scoring='accuracy', model_name='{config['model_name']}'\n"
        f")"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        f"# En iyi modeli değerlendir\n"
        f"best_results = evaluate_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test,\n"
        f"                             model_name='{config['model_name']} (Optimized)')"
    ))
    
    # Model kaydetme
    section_num += 1
    cells.append(nbf.v4.new_markdown_cell(f"## {section_num}. Model Kaydetme"))
    cells.append(nbf.v4.new_code_cell(
        f"# En iyi modeli kaydet\n"
        f"save_model(best_model, '{config['save_name']}')\n\n"
        f"print('✓ {config['title']} modeli başarıyla kaydedildi!')"
    ))
    
    nb['cells'] = cells
    
    # Notebook'u kaydet
    with open(config['filename'], 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"✓ {config['filename']} oluşturuldu!")


if __name__ == '__main__':
    print("Model notebook'ları oluşturuluyor...\n")
    
    for config in models_config:
        create_model_notebook(config)
    
    print("\n✅ Tüm model notebook'ları başarıyla oluşturuldu!")


