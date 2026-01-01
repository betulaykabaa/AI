# AI# AI & Machine Learning Learning Path 🚀

Bu repo, sıfırdan başlayarak veri bilimi, istatistiksel analiz, veri ön işleme ve makine öğrenmesi algoritmalarını (Regresyon & Sınıflandırma) içeren kapsamlı bir öğrenme yolculuğunun kaynak kodlarını barındırır.

Her dosya, belirli bir konsepti izole bir şekilde ele alır ve görselleştirme ile destekler.

## Dosya Listesi ve İçerikleri

### 1. İstatistik, Olasılık ve Temeller
Yapay zekanın matematiksel altyapısı.
* **01-The-Bell-Curve.py** - Normal Dağılım (Gauss Eğrisi) ve standart sapmanın görselleştirilmesi.
* **02-Histogram-vs-Theory.py** - Gerçek veri histogramı ile teorik olasılık eğrisinin karşılaştırılması.
* **03-Standard-Scaler.py** - Veri ölçeklendirme (Standardization/Z-Score) ve dağılıma etkisi.
* **04-Bernoulli.py** - Bernoulli dağılımı (Yazı/Tura mantığı) simülasyonu.
* **05-Dropout.py** - Sinir ağlarında kullanılan "Dropout" (Unutma/Seyreltme) tekniğinin mantığı.
* **06-Bayes-Theorem-Cancer.py** - Bayes Teoremi ile olasılık güncelleme (Hastalık testi örneği).
* **07-Naive-Bayes.py** - Naive Bayes sınıflandırıcısının temel olasılık hesaplaması.
* **08-T-Test.py** - İki model veya grup arasındaki farkın istatistiksel anlamlılığı (T-Testi).
* **09-Tail-Test.py** - Hipotez testlerinde P-Value, Kritik Bölgeler ve Kuyruk (Tail) analizi.
* **10-Analysis-of-Variance.py** - Değişkenlerin hedef üzerindeki etkisini ölçen ANOVA testi.

### 2. Veri Ön İşleme ve Görselleştirme Teknikleri
Ham veriyi modele hazırlama araçları.
* **11-Label-One-Hot-Encoding.py** - Kategorik (String) verileri sayısal formata çevirme (Encoding).
* **12-Train_Test_Split.py** - Verisetini Eğitim (Train) ve Test olarak bölme stratejisi.
* **13-Reashape.py** - Numpy dizilerini (Array) model girişine uygun hale getirme (Reshaping).
* **14-Alpha.py** - Grafiklerde saydamlık (Alpha) kullanarak veri yoğunluğunu görme.
* **15-Edgecolor-s.py** - Scatter plot grafiklerinde nokta boyutu ve kenarlık stilleri.
* **16-Scatter.py** - İki değişken arasındaki ilişkiyi gösteren temel saçılım grafiği.
* **17-Describe.py** - Pandas ile verinin istatistiksel özetini (Mean, Std, Min, Max) çıkarma.
* **18-Corr.py** - Özellikler arası ilişkiyi ölçen Korelasyon Matrisi ve Isı Haritası.
* **19-Fit-Predict.py** - Scikit-learn kütüphanesinin standart eğitim akışı (.fit / .predict).

### 📈 3. Regresyon (Sayı Tahmini) Algoritmaları
Sürekli sayısal değerleri tahmin etme modelleri.
* **20-Lineer-Regression.py** - Basit Doğrusal Regresyon (Tek değişken ile tahmin).
* **21-Polynomial-regression.py** - Polinom Regresyon (Doğrusal olmayan/kıvrımlı veriler).
* **22-Multiple- Regression.PY** - Çok Değişkenli Regresyon (Birden fazla kriterle fiyat tahmini).
* **23-ridge-lasso.py** - Ridge (L2) ve Lasso (L1) regularizasyonu ile aşırı öğrenmeyi (Overfitting) engelleme.
* **24-Comparison.py** - Linear, Ridge ve Lasso modellerinin gürültülü veride karşılaştırılması.

### 4. Sınıflandırma (Classification) Algoritmaları
Veriyi sınıflara ayırma (0/1, Kedi/Köpek) modelleri.
* **25-Logistic-Regression.py** - Lojistik Regresyon ve Sigmoid fonksiyonu ile sınıflandırma.
* **26-svm-rbf.py** - Destek Vektör Makineleri (SVM) ve RBF Kernel (Halka şeklinde veri ayrımı).
* **27-KNN.py** - K-En Yakın Komşu (KNN) algoritması ve gürültülü verideki davranışı.
* **28-The-Clash-of-Classifiers.py** - Logistic Reg, SVM ve KNN modellerinin aynı arenada kapışması.
* **29-Gaussion-Naive-Bayes.py** - Sürekli veriler (Örn: Sağlık verileri) için Gaussian Naive Bayes.
* **30-multinominal-bernoulli-nb.py** - Metin madenciliği ve spam filtresi için Naive Bayes türleri.
* **31-Heatmap.py** - **Eksik Veri Analizi:** Eksik (NaN) verilerin tespiti, görselleştirilmesi ve doldurulması (Imputation).
* **32-Decision-Tree.py** - Karar Ağacı algoritması, kural görselleştirme ve ağaç yapısının çizimi.
* **33-Random-Forest.py** - Rastgele Orman algoritması ve Özellik Önem Düzeyi (Feature Importance).
* **34-LDA.py** - Lineer Diskriminant Analizi (LDA) ve özelliklerin sınıflara etkisi (Coefficients).
* **35-QDA.py** - Karesel Diskriminant Analizi (QDA) ile kıvrımlı karar sınırları çizimi.
* **36-Ada-Bost-Classifier.py** - AdaBoost algoritması: Zayıf öğrenicilerin birleşerek karmaşık şekilleri (Daire/Kare) öğrenmesi.

---
*Bu proje, Yapay Zeka öğrenim sürecim kapsamında oluşturulmuştur.*

Linear Regression (Doğrusal Regresyon): En temel, en basit algoritmadır.
Polynomial Regression: Doğrusal olmayan (kıvrımlı) sayısal veriler içindir.
Ridge & Lasso & ElasticNet: Bunlar Linear Regression'ın "Overfitting" (ezberleme) yapmasını engelleyen, cezalandırma yöntemli versiyonlarıdır. 
Naive Bayes Ailesi: (Gaussian, Multinomial, Bernoulli) - Olasılık temelli.
Decision Tree (Karar Ağacı): Soru-cevap temelli.
Random Forest: Çoğunluk oyu (Bagging) temelli.
KNN (K-Nearest Neighbors): Mesafe/Komşuluk temelli.
Logistic Regression: Sınıflandırma (0/1) için çizgi çizme.
SVM (Support Vector Machines): En geniş sınır çizgisi.
LDA (Linear Discriminant Analysis): Sınıf ayırmak için boyut indirgeme.
AdaBoost: Hatalara odaklanan ilk boosting algoritması.
GBM (Gradient Boosting Machine): Hataları matematiksel (türev) olarak düzelten yapı.
XGBoost / LightGBM / CatBoost: Bunlar şu an endüstri standardıdır.
QDA (Quadratic Discriminant Analysis): LDA'nın kardeşidir. LDA düz çizgi çizerken, QDA kıvrımlı sınırlar çizebilir.
Multi-Layer Perceptron (MLP): Bu aslında "Yapay Sinir Ağları"nın (Deep Learning) başlangıcıdır. Ama scikit-learn içinde supervised bir model olarak da kullanılır.
