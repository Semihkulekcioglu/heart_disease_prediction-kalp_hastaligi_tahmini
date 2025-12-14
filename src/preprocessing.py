"""
Veri Ön İşleme Fonksiyonları
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(file_path):
    """
    Veri setini yükler
    
    Parameters:
    -----------
    file_path : str
        Veri seti dosya yolu
    
    Returns:
    --------
    df : DataFrame
        Yüklenen veri seti
    """
    df = pd.read_csv(file_path)
    print(f"✓ Veri seti yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    return df


def check_missing_values(df):
    """
    Eksik değerleri kontrol eder
    
    Parameters:
    -----------
    df : DataFrame
        Kontrol edilecek veri seti
    
    Returns:
    --------
    missing_info : DataFrame
        Eksik değer bilgileri
    """
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_info = pd.DataFrame({
        'Eksik Değer Sayısı': missing_count,
        'Eksik Değer Yüzdesi (%)': missing_percent
    })
    missing_info = missing_info[missing_info['Eksik Değer Sayısı'] > 0].sort_values(
        'Eksik Değer Sayısı', ascending=False
    )
    
    if len(missing_info) == 0:
        print("✓ Veri setinde eksik değer bulunmamaktadır.")
    else:
        print(f"⚠ Toplam {len(missing_info)} sütunda eksik değer bulundu:")
        print(missing_info)
    
    return missing_info


def handle_missing_values(df, strategy='mean'):
    """
    Eksik değerleri işler
    
    Parameters:
    -----------
    df : DataFrame
        İşlenecek veri seti
    strategy : str, default='mean'
        Eksik değer doldurma stratejisi ('mean', 'median', 'most_frequent')
    
    Returns:
    --------
    df : DataFrame
        İşlenmiş veri seti
    """
    if df.isnull().sum().sum() == 0:
        return df
    
    imputer = SimpleImputer(strategy=strategy)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    print(f"✓ Eksik değerler '{strategy}' stratejisi ile dolduruldu.")
    return df_imputed


def check_outliers(df, columns):
    """
    Aykırı değerleri IQR yöntemi ile tespit eder
    
    Parameters:
    -----------
    df : DataFrame
        Kontrol edilecek veri seti
    columns : list
        Kontrol edilecek sütunlar
    
    Returns:
    --------
    outlier_info : dict
        Aykırı değer bilgileri
    """
    outlier_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_info[col] = {
                'count': outlier_count,
                'percentage': (outlier_count / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    
    if outlier_info:
        print(f"⚠ Toplam {len(outlier_info)} sütunda aykırı değer bulundu:")
        for col, info in outlier_info.items():
            print(f"  - {col}: {info['count']} aykırı değer ({info['percentage']:.2f}%)")
    else:
        print("✓ Belirtilen sütunlarda aykırı değer bulunmadı.")
    
    return outlier_info


def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Veri setini eğitim ve test olarak böler
    
    Parameters:
    -----------
    df : DataFrame
        Bölünecek veri seti
    target_column : str
        Hedef değişken adı
    test_size : float, default=0.2
        Test seti oranı
    random_state : int, default=42
        Rastgelelik için seed değeri
    
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Bölünmüş veri setleri
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"✓ Veri seti bölündü:")
    print(f"  - Eğitim seti: {X_train.shape[0]} örnek")
    print(f"  - Test seti: {X_test.shape[0]} örnek")
    print(f"  - Özellik sayısı: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test, method='standard'):
    """
    Özellikleri ölçeklendirir
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Eğitim verisi
    X_test : DataFrame or array
        Test verisi
    method : str, default='standard'
        Ölçeklendirme yöntemi ('standard', 'minmax', 'robust')
    
    Returns:
    --------
    X_train_scaled, X_test_scaled : arrays
        Ölçeklendirilmiş veri setleri
    scaler : scaler object
        Kullanılan scaler nesnesi
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
    else:
        raise ValueError(f"Geçersiz ölçeklendirme yöntemi: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Özellikler '{method}' yöntemi ile ölçeklendirildi.")
    
    return X_train_scaled, X_test_scaled, scaler


def get_feature_info(df):
    """
    Veri seti hakkında özet bilgi verir
    
    Parameters:
    -----------
    df : DataFrame
        İncelenecek veri seti
    
    Returns:
    --------
    info_dict : dict
        Veri seti bilgileri
    """
    info_dict = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(exclude=[np.number]).columns.tolist()
    }
    
    print("=" * 60)
    print("VERİ SETİ BİLGİLERİ")
    print("=" * 60)
    print(f"Boyut: {info_dict['shape'][0]} satır x {info_dict['shape'][1]} sütun")
    print(f"Sayısal sütunlar: {len(info_dict['numeric_columns'])}")
    print(f"Kategorik sütunlar: {len(info_dict['categorical_columns'])}")
    print(f"Toplam eksik değer: {sum(info_dict['missing_values'].values())}")
    print("=" * 60)
    
    return info_dict


