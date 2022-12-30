# ANALISIS-SENTIMEN-PADA-PERPINDAHAN-IBUKOTA-INDONESIA-DENGAN-ALGORITMA-SUPPORT-VECTOR-MACHINE
Analysis of the opinions expressed on Twitter regarding the relocation of Indonesia's capital city using combination of algorithm classifiers Support Vector Machine (SVM), Feature Selection Term Frequency Inverse Document (TF-IDF), and Bag of Words, and also using a Lexicon-based approach for labeling data as positive or negative sentiment

## Data Input
- **Dataset:** Twitter Data regarding the relocation of Indonesia's capital city (May-September 2022)
- **Slang & Stop Words:** [Colloquial Indonesian Lexicon](https://github.com/nasalsabila/kamus-alay) dan [ID-Stopwords](https://github.com/stopwords-iso/stopwords-id);
- **Leksikon:** [InSet](https://github.com/fajri91/InSet) dan sentiwords_id (dari [sentistrength_id](https://github.com/masdevid/sentistrength_id));
- **Ekstraksi Fitur:**`bag of words`, `TF-IDF`;
- **Hyperparameter Tunning:** `Grid Search Cross Validation`
- **Classifier Algorithm:** `SVM` dengan *Linear, RBF, Polynomial Karnel*
- ** Evaluation:** `K-Fold Cross Validation` and `Confusion Matrix`

## Library Python
- Pandas, Numpy, Tweepy,  Sci-Kit Learn, NLTK, Matplotlib


## Flowchart
- **Craawling Data** menggunakan `Twitter API`
- Data text yang telah diambil akan dilakukan proses `Data Preprocessing` dengan Tokenizing, remove punctuation (ReGex), casefolding, remove duplicate tweet, filtering, stemming, stopword.
- Data teks yang diambil akan diberi label sentimen positif dan negatif menggunakan lexicon InSet dan SentiStrength.
- Pembobotan kata dengan ekstraksi fitur `TF-IDF` dan `BoW`
- `Hyperparameter tunning` menggunakan `Grid Search Cross Validation` untuk mendapatkan kombinasi kernel dan parameter terbaik dengan 3 kernel uji yaitu kernel linear dengan nilai cost={1, 100, 1000}, kernel Radial Basis Function (RBF) dengan parameter cost={1, 100, 1000}, gamma={0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0,9}, dan kernel polynomia dengan parameter cost={1, 100, dan 100}, gamma={0.01, 0.002, 0.003, 0.004, 0.05} dan degree={2,3,4}
- Klasifikasi sentimen menggunakan algoritma klasifikasi `SVM` dengan `parameter terbaik hasil Grid Search Cross Validation`.
- Menggunakan metode `K-Fold Cross Validation` dengan 5 iterasi dan `confussion matrix` untuk mengevaluasi keakuratan model prediksi.

![skenario pengujian umum](https://user-images.githubusercontent.com/55600482/210097901-c034dcc6-988c-49eb-a22c-54598754d3af.png)
### Skenario 1
![skenario 1](https://user-images.githubusercontent.com/55600482/210097899-544aa5e2-0070-41aa-bc77-69f49832d06a.png)
### Skenarion 2
![skenario 2](https://user-images.githubusercontent.com/55600482/210097900-648e4570-f58f-49e8-a5ec-373f7b3f4fc5.png)

## Hasil
### Data Pre-processing
![Hasil Data preprocessing](https://user-images.githubusercontent.com/55600482/210098656-8bc3bc96-665b-4152-b82f-4312ea172b0f.png)

Sub-bab ini menjelaskan hasil dari proses crawling data dan data pre-processing. Proses crawling data dilakukan dari tanggal 17 Mei 2022 hingga September 2022 dengan topik perpindahan Ibukota Negara (IKN) Indonesia dan mendapatkan data tweet sebanyak 42.608 tweet dan setelah dilakukan pre-processing data didapatkan data bersih sebanyak 12.243 tweet

### Labelling
![hasil labelling 1](https://user-images.githubusercontent.com/55600482/210098651-8d236124-7421-4653-98a3-7b5de09ff2ef.JPG)
![hasil labelling diagram](https://user-images.githubusercontent.com/55600482/210098653-d3c9a367-b36e-496e-882d-27bd194e8094.JPG)

Hasil dari analisis sentimen mengenai IKN pada media sosial Twitter menunjukkan 59,32% dari 12.243 tweet bersentimen positif menggunakan lexicon InSet, sedangkan menggunakan lexicon SentiStrength didapatkan 58,73% dari 12.243 tweet bersentimen positif. Dari kedua data tersebut dapat dilihat bahwa pendapat publik cenderung positif atau mendukung adanya perpindahan Ibu Kota Negara Indonesia ke Kalimantan Timur

### Hyperparameter Tunning dengan Grid Search Cross Validation
Hasil dari Hyperparameter Tunning menggunakan Grid Search Cross Validation dengan 5 iterasi (K=5) pada masing-masing data uji kombinasi ekstraksi fitur dan labelling lexicon.

![hasil hyperparameter tunning](https://user-images.githubusercontent.com/55600482/210098657-c657b649-f449-4dde-bea9-bd631e5037f7.JPG)

### Klasifikasi SVM dengan Parameter terbaik hasil Grid Search Cross validation
![hasil klasifikasi svm](https://user-images.githubusercontent.com/55600482/210098660-6740e577-0685-4124-bb45-ac09485ef16d.JPG)

Hasil perbandingan akurasi dari seluruh kombinasi ekstraksi fitur, algoritma SVM dengan seluruh hasil parameter terbaik Grid Cross Cross Validation pada masing-masing labelling lexicon di dapatkan best model dengan kombinasi algoritma SVM kernel linear (C = 1) dan ekstraksi fitur BoW pada hasil labelling lexicon SentiStrength dengan nilai akurasi sebesar 97,88% 

### Evaluasi Best Model
* Confusion Matrix

![matrix cm](https://user-images.githubusercontent.com/55600482/210099565-95828f35-93a0-4f22-b19a-b61ceef99080.JPG)

Gambar diatas menunjukkan 1462 + 2133 = 3595 merupakan prediksi kelas sentimen yang benar dan 29 + 49 = 78 prediksi kelas sentimen yang salah. Dalam hal ini, model memiliki:
  1.	True Positives (Actual Positive:1 dan Predict Positive:1) - 1462
  2. True Negatives (Actual Negative:0 dan Predict Negative:0) - 2113
  3.	False Positives (Actual Negative:0 tetapi Predict Positive:1) - 29 (Kesalahan tipe I)
  4.	False Negatives Actual Positive:1 tetapi Predict Negative:0) - 49 (Kesalahan Tipe II)

* Clasification Report

![cm evaluation](https://user-images.githubusercontent.com/55600482/210099493-1b21768e-85de-4de5-87f7-4bf6a3b6bae1.JPG)

Classification report pada best model yaitu kombinasi SVM kernel linear (C=1) dan ekstraksi fitur BoW dengan hasil labelling data SentiStrength didapatkan nilai precision, recall, f1-score sebesar 0,98 dimana Nilai terbaik F1-Score adalah 1.0 dan nilai terburuknya adalah 0. Sehingga kinerja klasifikasi menggunakan best model yang digunakan sangat baik.

* 5-Fold Cross Validation

Hasil rata-rata dari evaluasi best model menggunakan 5-fold cross validation. Model yang telah di uji sebanyak 5 kali dengan data latih dan uji yang telah di shuffle mendapatkan hasil rata rata 0,98 untuk nilai precision, recall dan F1-score. Sedangkan nilai rata-rata accuracy sebesar 0,97.

![rerata k-fold cross validation](https://user-images.githubusercontent.com/55600482/210099495-8d89833c-92ce-49c4-9ce2-9b95f89585fe.JPG)
