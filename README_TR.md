# 🫀 Kalp Hastalığı Tahmini - ML Sınıflandırma Projesi

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Türkçe** | [English](README.md)

Kalp hastalığı tahmini için 6 farklı makine öğrenmesi algoritmasının karşılaştırıldığı kapsamlı bir proje.

## 📊 Proje Hakkında

Bu projede kalp hastalığı tahmini için **6 farklı makine öğrenmesi algoritması** geliştirilmiş ve karşılaştırılmıştır:

- ✅ Logistic Regression (Lojistik Regresyon)
- ✅ k-Nearest Neighbors (k-En Yakın Komşu)
- ✅ Decision Tree (Karar Ağacı)
- ✅ Random Forest (Rastgele Orman)
- ✅ LightGBM
- ✅ XGBoost

## 🎯 Özellikler

- **Keşifsel Veri Analizi (EDA)** ile kapsamlı görselleştirmeler
- **6 ML algoritması** ve hiperparametre optimizasyonu
- **Performans karşılaştırması** tüm metriklerle
- **Yeniden kullanılabilir kod** modüler yapıda
- **Profesyonel dokümantasyon** ve temiz kod

<p align="center">
  <img src="https://github.com/user-attachments/assets/1a92fbcf-5dc2-4170-83f1-45089fa98ae1" width="400" />
  <img src="https://github.com/user-attachments/assets/7667bf29-cbcf-4e19-b2d1-ec50b63132da" width="400" />
  <img src="https://github.com/user-attachments/assets/11b2a060-de8d-4385-a0d5-67047547c93e" width="600" />
</p>

## 📁 Proje Yapısı

```
├── data/                      # Veri seti
│   └── heart_disease.csv     # Kalp hastalığı verisi (303 hasta)
├── src/                       # Python modülleri
│   ├── preprocessing.py      # Veri ön işleme fonksiyonları
│   └── model_utils.py        # Model yardımcı fonksiyonları
├── notebooks/                 # Jupyter notebook'lar
│   ├── 01_veri_analizi.ipynb           # EDA
│   ├── 02_logistic_regression.ipynb    # Lojistik Regresyon
│   └── 08_model_karsilastirma.ipynb   # Model Karşılaştırması ⭐
├── models/                    # Kaydedilmiş modeller
├── results/                   # Sonuçlar ve görseller
└── requirements.txt          # Gerekli kütüphaneler
```

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repoyu klonlayın
git clone https://github.com/Semihkulekcioglu/heart_disease_prediction-kalp_hastaligi_tahmini.git
cd heart_disease_prediction-kalp_hastaligi_tahmini

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt

# Jupyter Notebook'u başlatın
jupyter notebook
```

### Kullanım

**Önerilen:** `notebooks/08_model_karsilastirma.ipynb` dosyasını çalıştırarak 6 modeli aynı anda eğitin ve karşılaştırın!

## 📈 Model Performansları

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.82     | 0.83      | 0.80   | 0.81     | 0.88    |
| k-NN                | 0.85     | 0.84      | 0.86   | 0.85     | 0.90    |
| Decision Tree       | 0.78     | 0.75      | 0.82   | 0.78     | 0.80    |
| Random Forest       | 0.88     | 0.89      | 0.87   | 0.88     | 0.93    |
| LightGBM            | 0.90     | 0.91      | 0.89   | 0.90     | 0.95    |
| XGBoost             | 0.89     | 0.90      | 0.88   | 0.89     | 0.94    |

🏆 **En İyi Model:** LightGBM - %90 doğruluk ve 0.95 ROC-AUC

## 📊 Veri Seti

**Kalp Hastalığı Veri Seti** 303 hasta kaydı ve 14 özellik içerir:

- Yaş, cinsiyet, göğüs ağrısı tipi
- Tansiyon, kolesterol
- EKG sonuçları
- Maksimum kalp atış hızı
- Egzersiz anjinası, ST depresyonu
- Hedef: Hastalık varlığı (0=sağlıklı, 1=hasta)

## 🛠️ Teknolojiler

- **Python 3.8+**
- **Scikit-learn** - ML algoritmaları
- **Pandas & NumPy** - Veri işleme
- **Matplotlib & Seaborn** - Görselleştirme
- **XGBoost & LightGBM** - Gradient boosting
- **Jupyter Notebook** - İnteraktif geliştirme

## 📝 Öğrenilenler

- ✅ Keşifsel Veri Analizi (EDA)
- ✅ Veri ön işleme ve ölçeklendirme
- ✅ 6 farklı sınıflandırma algoritması
- ✅ Hiperparametre optimizasyonu (GridSearchCV)
- ✅ Model değerlendirme metrikleri
- ✅ Model karşılaştırma teknikleri

## 🎓 Eğitim Değeri

Şunlar için idealdir:
- Makine öğrenmesi başlangıç seviyesi
- Veri bilimi öğrencileri
- Portfolio projeleri
- Kaggle yarışması pratiği

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Şunları yapabilirsiniz:
- Hata bildirimi
- Özellik önerisi
- Pull request gönderimi

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.
