# Diabetes Prediction Menggunakan Machine Learning - Alif Khusain Bilfaqih

## 1. Domain Proyek
Proyek ini dibuat dalam bidang kesehatan, yaitu tentang penyakit diabetes yang sudah sangat banyak terjadi, baik di Indonesia maupun di dunia.

### Latar Belakang

Diabetes adalah kondisi kronis yang terjadi saat pankreas tidak dapat lagi memproduksi insulin, atau tubuh tidak dapat menggunakan insulin secara efektif.
Insulin adalah hormon yang dibuat oleh pankreas yang berfungsi sebagai kunci untuk menyalurkan glukosa dari makanan yang kita makan dari aliran darah ke dalam sel-sel tubuh untuk menghasilkan energi. Tubuh memecah semua makanan karbohidrat menjadi glukosa dalam darah, dan insulin membantu glukosa bergerak ke dalam sel-sel.
Bila tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif, hal ini menyebabkan kadar glukosa darah tinggi, yang disebut hiperglikemia
<sup>[1](https://idf.org/about-diabetes/what-is-diabetes/)</sup>.

Menurut data tahun 2021 yang dilansir dari situs resmi IDF (International Diabetes Federation), 10,5% atau 537 juta orang dewasa di dunia (Umur 20-79 tahun) hidup dengan mengidap diabetes. Angka ini diprediksi akan terus meningkat hingga ke angka 643 juta di tahun 2030, dan 783 juta di tahun 2045.
Sedangkan jumlah kematian yang disebabkan oleh diabetes pada tahun 2021 terjadi sebanyak 6,7 juta (1 kasus kematian setiap 5 detiknya)
<sup>[2](https://idf.org/about-diabetes/diabetes-facts-figures/)</sup>
<sup>[3](https://diabetesatlas.org/#:~:text=Diabetes%20around%20the%20world%20in%202021%3A,%2D%20and%20middle%2Dincome%20countries.)</sup>.

Indonesia termasuk negara dengan jumlah diabetes terbanyak. Pada tahun 2021, Indonesia menempati peringkat kelima terbanyak dengan jumlah 19,5 juta penderita.
<sup>[4](https://idf.org/our-network/regions-and-members/western-pacific/members/indonesia/)</sup>.

Persoalan ini menjadi perhatian serius bagi Kementerian Kesehatan. Apalagi diabetes melitus merupakan ibu dari segala penyakit.

“Diabetes itu adalah mother of all diseases. Kalau tidak terkontrol, dia bisa terkena penyakit jantung, stroke, ginjal yang akan lebih berat lagi masalahnya, akan lebih berat lagi biayanya,” ujar Direktur Pencegahan dan Pengendalian Penyakit Tidak Menular Kementerian Kesehatan, Dr. Eva Susanti, S. Kp., M. Kes., kepada Mediakom pada Kamis, 14 Desember 2023
<sup>[5](https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis#:~:text=Menurut%20IDF%2C%20Indonesia%20menduduki%20peringkat,merupakan%20ibu%20dari%20segala%20penyakit.)</sup>.

Mengingat banyaknya kasus yang terjadi, proyek "Diabetes Prediction" ini dibuat untuk membantu orang orang mengantisipasi penyakit diabetes.

Hasil dari proyek ini adalah dashboard dan model prediksi
- Dashboard bertujuan untuk menghimbau orang tentang penyakit diabetes, seperti penyebab, gejala, dan cara pencegahan diabetes.
- Model prediksi yang dibuat dapat memprediksi apakah dia terkena diabetes atau tidak dengan data kesehatan yang dia milki, dan mengetahui rekomendasi hal apa yang perlu dilakukan untuk mengurangi resiko diabetes.

## 2. Business Understanding

### Problem Statements

Dari latar belakang yang sudah dijelaskan sebelumnya, dapat disimpulkan bahwa diabetes sudah menjadi ancaman serius bagi kesehatan saat ini. Oleh sebab itu, Proyek Diabetes Prediction ini dibuat.

Diantara beberapa masalah yang akan diselesaikan melalui Proyek Diabetes Prediction ini adalah sebagai berikut:
- Dari data kesehatan yang ada, apa faktor yang mempengaruhi orang terkena diabetes, dan korelasinya terhadap data kesehatan lainnya.
- Model yang dapat memprediksi apakah orang itu terkena diabetes atau tidak sesuai data kesehatannya, sehingga perlu mengetahui juga model dan algoritma apa yang terbaik.
- Hal apa saja yang sesuai untuk dia kerjakan agar dapat mengurangi resiko diabetes. Karena setiap orang memilki kondisi kesehatan yang berbeda yang mungkin salah satunya menjadi penyebab diabetes.

### Goals

Tujuan utama dari Proyek Diabetes Prediction ini adalah semua orang dapat mengantisipasi penyakit diabetes sedini mungkin. Terutama dengan adanya rekomendasi atau tips yang diberikan, harapannya ini dapat mengurangi penyakit diabetes bagi yang sudah terkena, dan mengurangi kondisi yang menyebabkan diabetes bagi yang tidak mengalaminya. Dan juga model yang dapat menentukan secara akurat apakah dia penderita diabetes atau tidak.

### Solution statements

- Untuk dapat menemukan model yang dapat memprediksi secara akurat, Proyek Diabetes Prediction ini akan melatih berbagai macam model, lalu memilihi model dengan akurasi paling tinggi sebagai model yang digunakan.
- Rekomendasi hal yang akan ditampilkan akan diambil dari beberapa sumber, yaitu:
- Memvisualisasikan semua fitur yang ada pada data dan mengkorelasikannya dengan fitur lainnya agar terlihat jelas apa saja yang dapat menyebabkan diabetes. Salah satu visualisasi yang akan dipakai yaitu heatmap dengan memanfaatkan library seaborn
- Untuk meningkatkan akurasi model, akan ditambahkan proses improvement pada machine learning yang dipilih dengan hyperparameter tuning.
- Metrik evaluasi yang digunakan adalah *accuracy*, *recall*, *precission*, dan *f1-score* yang ada dalam classification_report. Selain itu juga akan ada *confusion matrix*.

## 3. Data Understanding

Data yang digunakan dalam proyek ini merupakan data parameter kesehatan pasien tentang status diabetes mereka, yang di upload oleh Mohammed Mustafa di platform Kaggle.

Dataset yang dapat diunduh di: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) 

### Sample Data

|       | gender   |   age |   hypertension |   heart_disease | smoking_history   |   bmi |   HbA1c_level |   blood_glucose_level |   diabetes |
|------:|:---------|------:|---------------:|----------------:|:------------------|------:|--------------:|----------------------:|-----------:|
|     0 | Female   | 80    |              0 |               1 | never             | 25.19 |           6.6 |                   140 |          0 |
|     1 | Female   | 54    |              0 |               0 | No Info           | 27.32 |           6.6 |                    80 |          0 |
|     2 | Male     | 28    |              0 |               0 | never             | 27.32 |           5.7 |                   158 |          0 |
|     3 | Female   | 36    |              0 |               0 | current           | 23.45 |           5   |                   155 |          0 |
|     4 | Male     | 76    |              1 |               1 | current           | 20.14 |           4.8 |                   155 |          0 |

### Informasi Dasar Data

|       |         age |   hypertension |   heart_disease |          bmi |   HbA1c_level |   blood_glucose_level |      diabetes |
|:------|------------:|---------------:|----------------:|-------------:|--------------:|----------------------:|--------------:|
| count | 100000      |   100000       |   100000        | 100000       |  100000       |           100000      | 100000        |
| mean  |     41.8859 |        0.07485 |        0.03942  |     27.3208  |       5.52751 |              138.058  |      0.085    |
| std   |     22.5168 |        0.26315 |        0.194593 |      6.63678 |       1.07067 |               40.7081 |      0.278883 |
| min   |      0.08   |        0       |        0        |     10.01    |       3.5     |               80      |      0        |
| 25%   |     24      |        0       |        0        |     23.63    |       4.8     |              100      |      0        |
| 50%   |     43      |        0       |        0        |     27.32    |       5.8     |              140      |      0        |
| 75%   |     60      |        0       |        0        |     29.58    |       6.2     |              159      |      0        |
| max   |     80      |        1       |        1        |     95.69    |       9       |              300      |      1        |

Berikut adalah informasi dasar dari data yang akan digunakan:
- Format dataset berupa CSV (Comma-Seperated Values),
- Dataset memiliki 10.000 sample, dengan 8 *feature*,
- 2 *features* berupa string, dan lainnya berupa *float* atau *integer*,
- 3 *features* merupakan data kategorikal yang berupa 0 sebagai tidak, dan 1 sebagai iya, dan 2 *features* kategorikal dengan beberapa pilihan

### Variabel-variabel Diabetes Prediction Dataset

- gender: Merepresentasikan jenis kelamin pasien (Male: Pria, Female: Wanita, Other: Lainnya).
- age: Rentang usia pasien (Dari 0-80 tahun).
- hypertension: Kondisi kesehatan yang terjadi ketika tekanan darah sistolik (SBP) ≥ 140 mmHg dan/atau tekanan darah diastolik (DBP) ≥ 90 mmHg, menandakan memaksa jantung memompa darah lebih keras (0: Tidak ada *hypertension*, 1: Terdapat *hypertension*).
- heart_disease: Kondisi kesehatan lainnya yang dikaitkan dengan peningkatan risiko terkena diabetes (0: Tidak ada *heart disease*, 1: Terdapat *heart disease*).
- smoking_history: Riwayat pasien dalm hal merokok (not current: Saat ini tidak, former: Mantan perokok, No Info: Tidak ada keterangan apakah pernah atau tidak, current: Merokok sampai saat ini, never: Tidak pernah, Ever: Pernah).
- bmi: Body Mass Index adalah ukuran yang digunakan untuk memperkirakan jumlah lemak tubuh yang dihitung dengan membagi berat badan dalam kilogram dengan tinggi badan dalam meter kuadrat (m2). Rentang BMI dalam daatset ini yaitu 10.16 sampai 71.55 (Kekurangan berat badan: BMI kurang dari 18.5, Normal: 18.5-24.9, Kelebihan berat badan: 25-29.9 is overweight, Obesitas: 30 atau lebih).
- HbA1c_level: Level Hemoglobin A1c menunjukkan kadar gula darah rata-rata selama 2-3 bulan terakhir.
- blood_glucose_level: Merepresentasikan jumlah glukosa dalam aliran darah pada waktu tertentu. Inilah indikator dari diabetes.
- Diabetes: Menunjukan hasil diabetes (0: Tidak terindikasi diabeted, 1: Terindikasi diabetes).

### Exploratory Data Analysis

#### Age

Dapat dilihat pada hiplot di bawah ini, bahwa rentang umur dataset yaitu 0-80 tahun. Semakin bertambah usia, semakin tinggi pula pasien yang terkena diabetes. Artinya, umur cukup mempengaruhi diabetes.
![vis_1](images/vis_1.png)

#### Data Kategorikal
![categorical_visualization](images/vis_2.png)
Dari visualisasi dataset di atas, tidak ada yang terlalu menyebabkan orang terkena diabetes. Memang tingkat orang yang tidak terkena *hypertension* atau penyakit jantung lebih tinggi tidak terkena diabetes, namun jumlah data yang terkena *hypertension* atau penyakit jantung lebih sedikit. Begitu juga dengan jenis kelamin dan riwayat merokok.

#### Data Numerikal
![numerical_visualization](images/vis_4.png)
Berikut beberapa informasi yang dapat diambil:
- Orang dengan level Hemoglobin A1c di atas 6.5 dipastikan terkena diabetes. Namun ada beberapa orang dengan level Hemoglobin A1c di bawah 6.5 yang terkena diabetes.
- Orang dengan jumlah glukosa di atas 200 dipastikan terkena diabetes.
- Menurut data yang digunkakan, bmi atau jumlah perkiraan lemak pada tubuh tidak terlalu mempengaruhi diabetes, karena banyak juga pasien dengan bmi di atas 60 yang tidak terkena diabetes. 
- Kedua *features* ini adalah indikasi seseorang terkena diabetes atau tidak.

#### Heatmap

![heatmap_corr](images/heatmap_corr.png)
Di atas adalah heatmap korelasi/hubungan antara *feature* diabetes dengan fitur lainnya. Beberapa hasil yang didapat antara lain:
- 

## 4. Data Preparation

Berikut adalah urutan data preparation yang digunakan dalam proyek:

#### Missing Value

Mengetahui apakah ada data yang kosong pada setiap kolom menggunakan kode isnull(), dan menjumlahkan data yang bernilai koson tersebut dengan perintah sum().
Missnig value sering kali menjadi hambatan dan menurunkan tingkat akurasi, karena data yang hilang merupakan data penting yang juga dipelajari oleh model.

#### Duplikasi Data

Menyeleksi apakah ada data yang memiliki nilai yang sama dengan data lain. Untuk mencari data yang terduplikasi ini, dilakukan dengan menggunakan perintah duplicated().
Setelah itu, jumlah data yang sama ini akan ditotalkan dan menampilkan jumlahnya dengan perintah sum().

Duplikasi juga salah satu kesalahan data yang dapat mempengaruhi model, diantaranya:
- Data duplikasi biasanya tidak menambah informasi baru untuk model, sehingga keberadaannya tidak meningkatkan kemampuan model dalam mengenali pola.
- Memperbesar risiko overfitting, terutama untuk model yang sangat kompleks (seperti Decision Tree atau Random Forest). Hal ini karena model cenderung "mengingat" pola yang sering muncul secara berulang, meskipun pola tersebut berasal dari data yang sama.

Diketahui juga jika bahwa data non-diabetes sudah jauh lebih dominan.
Total data non-diabetes: 91500 - 3836 = 87664
Total data diabetes: 8500 - 18 = 8482
Rasio kategori setelah penghapusan yaitu 87664/8482 = 1034/1
Sehingga, menghapus data duplikasi ini tidak akan signifikan mempengaruhi distribusi dataset. Hal ini justru membuat dataset menjadi lebih bersih dan membantu mengurangi bias terhadap kategori non-diabetes.

Oleh karena itu, semua data yang terduplikasi akan dihapus, dan hanya menyisakan 1 data saja. Adapun perintah untuk menghapusnya secara otomatis yaitu denagn drop_duplicated(inplace=True).

#### Menghapus Nilai Yang Tidak Sesuai

Pada kolom 'gender', terdapat nilai 'Other', yaitu jenis kelamin selain Laki-laki dan perempuan. Karena hanya ada 2 jenis kelamin data dengan 'gender' 'Other' akan dihapus. Dan karena data ini hanya ada 18, maka tidak akan mempengaruhi nilai data.

Kolom smoking_history di sini akan dihapus, karena smoking_history memiliki banyak pilihan yang kurang jelas, dan mungkin akan mempersulit pembelajaran model.

#### Mengubah Tipe Data

Setelah memperhatikan semua data, terdapat tipe data string/object, yaitu 'gender' dan 'smoking_history'. Untuk memudahkan kita dalam menentukan penyebab diabetes, dan juga memudahkan proses pembelajaran mesin, data string ini akan diubah menjadi integer.

Untuk mengubah gender, di sini menggunakan perintah .cat.codes. Setelah dilihat menggunakan main_df.gender.cat.categories, dapat dilihat bahwa '0' mewakili Female/Perempuan, dan '1' mewaliki Male/Laki-laki

#### Menghapus Kolom Yang Tidak Penting

Kolom smoking_history di sini akan dihapus, karena smoking_history memiliki banyak pilihan yang kurang jelas, dan mungkin akan mempersulit pembelajaran model.

Selanjutnya, agar dapat memilih kolom mana yang penting, yang pertama yaitu mencari nilai korelasi setiap feature dengan nilai diabetes menggunakan peerintah corr(). Proses ini dapat menggunakan berbagai macam cara, diantaranya:
- Membuat heatmat dengan memanfaatkan library seaborn dan matplotlib
- Membuat dataset baru dengan nama 'top_features' yang berisikan features dengan nilai korelasi di atas 0.19
- Agar terlihat lebih jelas, nilai korelasi dataset 'top_features' dibuatkan grafik batang dengan judul 'Correlation of Features with Diabetes'

Dari tingkat korelasi yang ada, diketahui bahwa 5 features tertinggi yaitu 'blood_glucose_level', 'HbA1c_level', 'age', 'bmi', 'hypertension', sehingga hanya ini yang akan kita buat sebagai data.
Tahapan ini dilakukan agar data yang dilatih oleh model adalah data yang benar-benar berguna, sehingga menghasilkan akurasi lebih tinggi.

#### Membagi Train dan Test

Sebelum memasuki tahap modeling, data perlu dibagi menjadi data training dan data test. Data training merupakan data yang digunakan untuk membangun sebuah model dan mendapatkan bobot yang sesuai. Sedangkan data testing digunakan untuk mengetahui tingkat keakuratan hasil dengan nilai sebenarnya.

Sebelum dibagi menjadi train dan test, data yang sudah siap dibagi menjadi label/feature (pada data ini yaitu: 'diabblood_glucose_level', 'HbA1c_level', 'age', dan 'bmi') dan target (hasil yang ingin diprediksi, yaitu 'diabetes')
Cara mudah membagi data yaitu dengan menggunakan train_test_split. Berikut adalah kode untuk meng*import*nya 

``` from sklearn.model_selection import train_test_split ```

target_size atau rasio yang digunakan untuk proyek ini yaitu 0.2 (80% data training, 20% data train). Rasio ini umum digunakan untuk memberikan keseimbangan antara memiliki jumlah data yang cukup untuk melatih model dan menyediakan data yang cukup untuk menguji performa model.

#### Menambahkan `stratify=y` pada `train_test_split`

`stratify=y` berfungsi untuk menyeimbangkan pembagian/distribusi kelas agar seimbang antara data latih dan uji. Karena, jika salah satu kelas kurang terwakili dalam data latih, model mungkin tidak akan terlatih dengan baik, karena tidak mampu belajar pola dari kelas tersebut.

Contohnya, jika dataset asli memiliki 90% kelas non-diabetes (kelas mayoritas) dan 10% kelas diabetes (kelas minoritas), maka data latih dan uji masing-masing juga akan memiliki proporsi yang sama (90:10).

#### Menangani Dataset Tidak Seimbang

Seperti yang sudah ditunjukan pada info dataset sebelumnya (atau pada penjelasan dupliaced data), dataset ini memiliki jumlah pasien diabetes sebanyak 91500 dan pasien yang tidak terkena diabetes sebanyak 8500. Artinya, data ini sangat tidak seimbang. Kondisi ini akan berdampak pada hasil model yang keseluruhan mungkin terlihat memiliki akurasi tinggi, tetapi sebenarnya model hanya memprediksi dengan baik untuk kelas mayoritas, dan lain masalah lainnya. Disinilah metode SMOTE dan Undersampling digunakan.

Metode SMOTE (Synthetic Minority Oversampling Technique) adalah teknik untuk menangani ketidakseimbangan data dengan mensintesis sampel baru atau menciptakan sampel baru yang lebih bervariasi untuk kelas minoritas berdasarkan tetangga terdekat dari data yang ada. Implementasi SMOTE dilakukan dengan library `imbalanced-learn`. Adapun Undersampling adalah menghapus sebagian data dari kelas mayoritas untuk menyeimbangkan distribusi.

Pada proyek ini, akan menggabungkan kedua metode ini untuk memastikan distribusi data yang lebih seimbang dengan mengurangi ukuran kelas mayoritas dan memperkaya variasi kelas minoritas, yaitu dengan SMOTE-Tomek. Berikut adalah cara kerja SMOTE-Tomek:
- Pertama, SMOTE digunakan untuk meningkatkan jumlah data pada kelas minoritas dengan cara membuat data sintetis.
- Setelah itu, Tomek digunakan untuk menghapus data mayoritas yang sangat dekat dengan data minoritas. Hal ini bertujuan untuk mengurangi noise dan meningkatkan pemisahan antara kedua kelas.

## 5. Modeling

Tahap ini adalah tahap modeling, yaitu proses mesin mempelajari data yang sudah disiapkan. Pada proyek ini, dicoba menggunakan berbagai macam jenis algoritma machine learning, lalu mengambil model yang memiliki nilai paling tinggi. Algoritma yang dicoba adalah:
- LogisticRegression
- K-Nearest Neighbors
- Decision Tree
- Support Vector Machine(Linear Kernel)
- Support Vector Machine(Non-Linear Kernal)
- Neural Network
- Random Forest
- Gradient Boosting

Setelah dicoba dengan beragam algoritma machine learning, hasil *metric accuracy* menunjukan bahwa Gradient Boosting adalah algoritma terbaik. Oleh karena itu, proyek ini akan menggunakan Gradient Boosting.

### Gradient Boosting

####Cara Kerja Gradient Boosting

Algoritma Gradient Boosting bekerja dengan menggabungkan beberapa model yang lemah menjadi sebuah model yang lebih kuat. Model-model lemah ini sering disebut dengan weak learners, dan dapat berupa model regresi atau klasifikasi sederhana seperti Decision Tree.

Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi, Gradient Boosting akan menambahkan weak learner baru dan mengoreksi prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut.
Proses ini dilakukan secara berulang-ulang hingga model yang dihasilkan memenuhi kriteria tertentu, seperti nilai loss function yang cukup kecil.

#### Kelebihan dan Kekurangan Gradient Boosting

Berikut adalah kelebihan algoritma Gradien Boosting:
- Gradient Boosting sering menghasilkan model yang akurat dan kuat, terutama ketika digunakan pada data yang kompleks dan tidak terstruktur.
- Algoritma ini dapat digunakan pada berbagai jenis data tanpa memerlukan asumsi yang ketat, seperti asumsi tentang distribusi data atau homoskedastisitas.
Berikut adalah kekurangan algoritma Gradien Boosting:
- Algoritma ini memerlukan tuning parameter yang cermat untuk mendapatkan model yang optimal. Hal ini dapat memakan waktu dan mengharuskan penggunaan cross-validation dan teknik tuning parameter lainnya.
- Gradient Boosting dapat cenderung overfit pada data training jika tidak dilakukan pengaturan parameter yang baik. 
- Gradient Boosting memerlukan jumlah data yang besar untuk memperoleh model yang akurat dan stabil. Jika jumlah data terlalu sedikit, algoritma ini dapat menjadi tidak stabil dan menghasilkan model yang tidak akurat.

### Improvement dengan RandomizedSearchCV 

RandomizedSearchCV adalah salah satu teknik hyperparameter tuning untuk menemukan kombinasi parameter terbaik pada suatu model machine learning. Teknik ini bekerja secara acak mencoba berbagai kombinasi parameter yang ditentukan dalam suatu ruang pencarian. Dengan demikian, RandomizedSearchCV dapat membantu kita menemukan parameter optimal yang dapat meningkatkan kinerja model dalam waktu lebih singkat dibanding hyperparameter tuning lainnya.

Berikut adalah cara kerja RandomizedSearchCV:
- Menentukan rentang nilai atau distribusi probabilitas untuk setiap hyperparameter yang ingin dioptimalkan.
- Algoritma secara acak memilih kombinasi parameter dari ruang pencarian yang telah ditentukan.
- Model dilatih dengan menggunakan kombinasi parameter yang dipilih dan dievaluasi menggunakan metrik yang relevan (accuracy).
- Proses sampling dan evaluasi diulang sebanyak yang ditentukan, atau hingga mencapai kriteria berhenti yang telah ditetapkan.
- Setelah semua iterasi selesai, model dengan kinerja terbaik (berdasarkan metrik evaluasi) akan dipilih sebagai model akhir.

## 6. Evaluation

Metrik evaluasi yang digunakan adalah sebagai berikut:

### Accuracy
*Accuracy* adalah metric yang mengukur seberapa sering model machine learning memprediksi keluaran hasil secara tepat. Hasil akurasi ini adalah pembagian prediksi yang benar dengan jumlah seluruh data.

Hasil akurasi ini berada diantara angka 0 dan 1. Semakin besar nilai akurasi, maka semakin baik model yang dibuat. Artinya, jika hasil accuracy bernilai 1, maka model mampu memprediksi seluruh data tanpa satupun kesalahan.

Untuk menghitung nilai akurasi, menggunakan rumus berikut:
![accuracy](images/accuracy.png)
- TP = True Positive (Model memprediksi dengan benar bahwa data bernilai positive)
- TN = True Negative (Model memprediksi dengan benar bahwa data bernilai negative)
- FP = False Positive (Model memprediksi dengan salah bahwa data bernilai positive)
- FN = False Negative (Model memprediksi dengan salah bahwa data bernilai negative)

### Recall
Recall/Sensitivitas adalah metrik evaluasi yang menggambarkan seberapa baik suatu model dalam mengidentifikasi positif dengan benar.

Sebagai analogi, bayangkan kita sedang mencari jarum di tumpukan jerami. Recall menggambarkan seberapa baik kita menemukan semua jarum yang ada di tumpukan tersebut. Jika kita menemukan 6 dari 10 dari jarum ditumpukan Jerami tersebut, artinya kita masih melewatkan 4 jarum yang belum ditemukan.

Dalam proyek ini, recall ini sangat penting, karena biaya salah mendiagnosis pasien positif (false negative) lebih tinggi. Contoh, pasien yang seharusnya menjalani pengobatan karena diabetes malah memakan makanan manis yang memperburuk penyakit diabetesnya.

Untuk menghitung recall, Anda dapat membagi jumlah true positive dengan jumlah contoh positif. Semakin tinggi recall, semakin banyak positif yang terdeteksi.
![recall](images/recall.png)

### Precision
Precision/Presisi adalah metrik yang mengukur seberapa sering model pembelajaran mesin memprediksi kelas positif dengan benar.
Presisi dihitung dengan membagi jumlah prediksi positif yang benar (positif benar) dengan jumlah total prediksi positif (positif benar ditambah positif salah).
![precision](images/precision.png)

#### Hasil Metrik precision, recall, dan f1-score

              precision    recall  f1-score   support

           0       0.97      1.00      0.98     17516
           1       1.00      0.67      0.80      1710

    accuracy                           0.97     19226
   macro avg       0.98      0.83      0.89     19226
weighted avg       0.97      0.97      0.97     19226

Berdasarkan metrik yang diberikan, kita dapat menyimpulkan hal berikut:
- Model ini sangat tepat dalam mengidentifikasi contoh kelas 0.
- Model ini hampir sempurna dalam mengingat contoh kelas 0.
- Skor F1 juga sangat tinggi, yang menunjukkan kinerja keseluruhan yang kuat untuk kelas 0.
- Model ini sempurna dalam presisi untuk kelas 1.
- Namun, model ini relatif lemah dalam mengingat contoh kelas 1.
- Skor F1 sedang, yang menunjukkan keseimbangan antara presisi dan mengingat, tetapi lebih rendah daripada kelas 0.

#### Kinerja Keseluruhan

- Keakuratan keseluruhan tinggi, yang menunjukkan bahwa model ini secara umum akurat.
- Skor F1 rata-rata makro sedang, yang menunjukkan bahwa model berkinerja baik secara rata-rata di kedua kelas.
- Skor F1 rata-rata tertimbang tinggi, yang menunjukkan bahwa kinerja model sangat dipengaruhi oleh kelas mayoritas (kelas 0). Kesimpulan

#### Kesimpulan

Model ini bekerja sangat baik pada kelas mayoritas (kelas 0) tetapi kesulitan pada kelas minoritas (kelas 1). Ini adalah masalah umum dalam kumpulan data yang tidak seimbang. Untuk meningkatkan kinerja pada kelas minoritas, teknik seperti oversampling, undersampling, atau pembobotan kelas dapat dipertimbangkan.

### Confusion Matrix

Confusion matrix adalah sebuah tabel yang digunakan untuk mengevaluasi performa model klasifikasi. Tabel ini membandingkan prediksi model dengan nilai aktual (ground truth) dari data. Dengan kata lain, confusion matrix menunjukkan seberapa baik model kita dalam mengklasifikasikan data ke dalam kelas yang benar.

#### Hasil Confusion Matrix

![confusion_matrix](images/confusion_matrix.png)
Ini adalah *confusion matrix* dari hasil model yang dibuat. Berikut adalah penjelasannya:
- (0,0): Model memprediksi kelas 0 dan benar (True Negative). Ada 17516 data yang diklasifikasikan dengan benar sebagai kelas 0.
- (0,1): Model memprediksi kelas 1, padahal sebenarnya kelas 0 (False Positive). Ada 0 data yang salah diklasifikasikan sebagai kelas 1.
- (1,0): Model memprediksi kelas 0, padahal sebenarnya kelas 1 (False Negative). Ada 567 data yang salah diklasifikasikan sebagai kelas 0.
- (1,1): Model memprediksi kelas 1 dan benar (True Positive). Ada 1143 data yang diklasifikasikan dengan benar sebagai kelas 1.

#### Kesimpulan dari Confusion Matrix ini:

Model sangat baik dalam mengklasifikasikan kelas 0. Hampir semua data kelas 0 berhasil diprediksi dengan benar.
Akan tetapi, model kurang baik dalam mengklasifikasikan kelas 1. Ada cukup banyak data kelas 1 yang salah diklasifikasikan sebagai kelas 0 (False Negative). Ini mengindikasikan bahwa model mungkin kesulitan dalam mendeteksi kelas minoritas (jika kelas 1 adalah kelas minoritas).
