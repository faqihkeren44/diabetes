# Diabetes Prediction Menggunakan Machine Learning - Alif Khusain Bilfaqih

## 1. Domain Proyek
Proyek ini dibuat dalam bidang kesehatan, yaitu tentang penyakit diabetes yang sudah sangat banyak terjadi, baik di Indonesia maupun di dunia.

### Latar Belakang

Diabetes adalah kondisi kronis yang terjadi saat pankreas tidak dapat lagi memproduksi insulin, atau tubuh tidak dapat menggunakan insulin secara efektif.
Insulin adalah hormon yang dibuat oleh pankreas yang berfungsi sebagai kunci untuk menyalurkan glukosa dari makanan yang kita makan dari aliran darah ke dalam sel-sel tubuh untuk menghasilkan energi. Tubuh memecah semua makanan karbohidrat menjadi glukosa dalam darah, dan insulin membantu glukosa bergerak ke dalam sel-sel.
Bila tubuh tidak dapat memproduksi atau menggunakan insulin secara efektif, hal ini menyebabkan kadar glukosa darah tinggi, yang disebut hiperglikemia
<sup>[1](https://idf.org/about-diabetes/what-is-diabetes/)</sup>

Menurut data tahun 2021 yang dilansir dari situs resmi IDF (International Diabetes Federation), 10,5% atau 537 juta orang dewasa di dunia (Umur 20-79 tahun) hidup dengan mengidap diabetes. Angka ini diprediksi akan terus meningkat hingga ke angka 643 juta di tahun 2030, dan 783 juta di tahun 2045.
Sedangkan jumlah kematian yang disebabkan oleh diabetes pada tahun 2021 terjadi sebanyak 6,7 juta (1 kasus kematian setiap 5 detiknya)
<sup>[2](https://idf.org/about-diabetes/diabetes-facts-figures/)</sup>
<sup>[3](https://diabetesatlas.org/#:~:text=Diabetes%20around%20the%20world%20in%202021%3A,%2D%20and%20middle%2Dincome%20countries.)</sup>

Indonesia termasuk negara dengan jumlah diabetes terbanyak. Pada tahun 2021, Indonesia menempati peringkat kelima terbanyak dengan jumlah 19,5 juta penderita.
<sup>[4](https://idf.org/our-network/regions-and-members/western-pacific/members/indonesia/)</sup>
Persoalan ini menjadi perhatian serius bagi Kementerian Kesehatan. Apalagi diabetes melitus merupakan ibu dari segala penyakit.
“Diabetes itu adalah mother of all diseases. Kalau tidak terkontrol, dia bisa terkena penyakit jantung, stroke, ginjal yang akan lebih berat lagi masalahnya, akan lebih berat lagi biayanya,” ujar Direktur Pencegahan dan Pengendalian Penyakit Tidak Menular Kementerian Kesehatan, Dr. Eva Susanti, S. Kp., M. Kes., kepada Mediakom pada Kamis, 14 Desember 2023
<sup>[5](https://sehatnegeriku.kemkes.go.id/baca/blog/20240110/5344736/saatnya-mengatur-si-manis#:~:text=Menurut%20IDF%2C%20Indonesia%20menduduki%20peringkat,merupakan%20ibu%20dari%20segala%20penyakit.)</sup>

Mengingat banyaknya kasus yang terjadi, proyek "Diabetes Prediction" ini dibuat untuk membantu orang orang mengantisipasi penyakit diabetes.
Hasil dari proyek ini adalah dashboard dan model prediksi. Dashboard bertujuan untuk menghimbau orang tentang penyakit diabetes, seperti penyebab, gejala, dan cara pencegahan diabetes. Model prediksi yang dibuat dapat memprediksi apakah dia terkena diabetes atau tidak dengan data kesehatan yang dia milki, dan mengetahui rekomendasi hal apa yang perlu dilakukan untuk mengurangi resiko diabetes.

## 2. Business Understanding

### Problem Statements

Dari latar belakang yang sudah dijelaskan sebelumnya, dapat disimpulkan bahwa diabetes sudah menjadi ancaman serius bagi kesehatan saat ini. Oleh sebab itu, Proyek Diabetes Prediction ini dibuat. Diantara beberapa masalah yang akan diselesaikan melalui Proyek Diabetes Prediction ini adalah sebagai berikut:
- Dari data kesehatan yang ada, apa faktor yang mempengaruhi orang terkena diabetes, dan korelasinya terhadap data kesehatan lainnya.
- Model yang dapat memprediksi apakah orang itu terkena diabetes atau tidak sesuai data kesehatannya, sehingga perlu mengetahui juga model dan algoritma apa yang terbaik 
- Hal apa saja yang sesuai untuk dia kerjakan agar dapat mengurangi resiko diabetes. Karena setiap orang memilki kondisi kesehatan yang berbeda yang mungkin salah satunya menjadi penyebab diabetes.

### Goals

Tujuan utama dari Proyek Diabetes Prediction ini adalah semua orang dapat mengantisipasi penyakit diabetes sedini mungkin. Terutama dengan adanya rekomendasi atau tips yang diberikan, harapannya ini dapat mengurangi penyakit diabetes bagi yang sudah terkena, dan mengurangi kondisi yang menyebabkan diabetes bagi yang tidak mengalaminya. Dan juga model yang dapat menentukan secara akurat apakah dia penderita diabetes atau tidak.

### Solution statements

- Untuk dapat menemukan model yang dapat memprediksi secara akurat, Proyek Diabetes Prediction ini akan melatih berbagai macam model, lalu memilihi model dengan akurasi paling tinggi sebagai model yang digunakan.
- Rekomendasi hal yang akan ditampilkan akan diambil dari beberapa sumber, yaitu:
- Memvisualisasikan semua fitur yang ada pada data dan mengkorelasikannya dengan fitur lainnya agar terlihat jelas apa saja yang dapat menyebabkan diabetes. Salah satu visualisasi yang akan dipakai yaitu heatmap dengan memanfaatkan library seaborn

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:
- Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## 3. Data Understanding

Data yang digunakan dalam proyek ini merupakan data parameter kesehatan pasien tentang status diabetes mereka, yang di upload oleh Mohammed Mustafa di platform Kaggle.

Dataset yang dapat diunduh di: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) 

### Variabel-variabel pada Diabetes Prediction Dataset adalah sebagai berikut:
- gender: Merepresentasikan jenis kelamin pasien (Male: Pria, Female: Wanita, Other: Lainnya).
- age: Rentang usia pasien (Dari 0-80 tahun).
- hypertension: Kondisi kesehatan yang terjadi ketika tekanan darah sistolik (SBP) ≥ 140 mmHg dan/atau tekanan darah diastolik (DBP) ≥ 90 mmHg, menandakan memaksa jantung memompa darah lebih keras (0: Tidak ada *hypertension*, 1: Terdapat *hypertension*).
- heart_disease: Kondisi kesehatan lainnya yang dikaitkan dengan peningkatan risiko terkena diabetes (0: Tidak ada *heart disease*, 1: Terdapat *heart disease*).
- smoking_history: Riwayat pasien dalm hal merokok (not current: Saat ini tidak, former: Mantan perokok, No Info: Tidak ada keterangan apakah pernah atau tidak, current: Merokok sampai saat ini, never: Tidak pernah, Ever: Pernah).
- bmi: Body Mass Index adalah ukuran yang digunakan untuk memperkirakan jumlah lemak tubuh yang dihitung dengan membagi berat badan dalam kilogram dengan tinggi badan dalam meter kuadrat (m2). Rentang BMI dalam daatset ini yaitu 10.16 sampai 71.55 (Kekurangan berat badan: BMI kurang dari 18.5, Normal: 18.5-24.9, Kelebihan berat badan: 25-29.9 is overweight, Obesitas: 30 atau lebih).
- HbA1c_level: Level Hemoglobin A1c menunjukkan kadar gula darah rata-rata selama 2-3 bulan terakhir.
- blood_glucose_level: Merepresentasikan jumlah glukosa dalam aliran darah pada waktu tertentu. Inilah indikator dari diabetes.
- Diabetes: Menunjukan hasil diabetes (0: Tidak terindikasi diabeted, 1: Terindikasi diabetes).

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## 4. Data Preparation
Berikut adalah urutan data preparation yang digunakan dalam proyek:
#### Missing Value
Mengetahui apakah ada data yang kosong pada setiap kolom menggunakan kode isnull(), dan menjumlahkan data yang bernilai koson tersebut dengan perintah sum().
Missnig value sering kali menjadi hambatan dan menurunkan tingkat akurasi, karena data yang hilang merupakan data penting yang juga dipelajari oleh model.
#### Duplikasi Data
Menyeleksi apakah ada data yang memiliki nilai yang sama dengan data lain. Untuk mencari data yang terduplikasi ini, dilakukan dengan menggunakan perintah dupliacted().
Setelah itu, jumlah data yang sama ini akan ditotalkan dan menampilkan jumlahnya dengan perintah sum().
Duplikasi juga salah satu kesalahan yang dapat menyebabkan kesalahan interpretasi dan mengganggu analisis sehingga dapat berpotensi menyesatkan model pembelajaran.
Oleh karena iu, jika terdapat data yang terduplikasi, maka semua data itu akan dihapus, dan hanya menyisakan 1 data saja. Adapun perintah untuk menghapusnya secara otomatis yaitu denagn drop_duplicated(inplace=True).
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
from sklearn.model_selection import train_test_split. 
target_size atau rasio yang digunakan untuk proyek ini yaitu 0.2 (80% data training, 20% data train). Rasio ini umum digunakan untuk memberikan keseimbangan antara memiliki jumlah data yang cukup untuk melatih model dan menyediakan data yang cukup untuk menguji performa model.

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
**Cara Kerja Gradient Boosting**
Algoritma Gradient Boosting bekerja dengan menggabungkan beberapa model yang lemah menjadi sebuah model yang lebih kuat. Model-model lemah ini sering disebut dengan weak learners, dan dapat berupa model regresi atau klasifikasi sederhana seperti Decision Tree.
Algoritma ini menggunakan pendekatan iteratif, di mana setiap iterasi, Gradient Boosting akan menambahkan weak learner baru dan mengoreksi prediksi sebelumnya dengan memperhitungkan kesalahan pada prediksi tersebut.
Proses ini dilakukan secara berulang-ulang hingga model yang dihasilkan memenuhi kriteria tertentu, seperti nilai loss function yang cukup kecil.
**Kelebihan dan Kekurangan Gradient Boosting**
Berikut adalah kelebihan algoritma Gradien Boosting:
- Gradient Boosting sering menghasilkan model yang akurat dan kuat, terutama ketika digunakan pada data yang kompleks dan tidak terstruktur.
- Algoritma ini dapat digunakan pada berbagai jenis data tanpa memerlukan asumsi yang ketat, seperti asumsi tentang distribusi data atau homoskedastisitas.
Berikut adalah kekurangan algoritma Gradien Boosting:
- Algoritma ini memerlukan tuning parameter yang cermat untuk mendapatkan model yang optimal. Hal ini dapat memakan waktu dan mengharuskan penggunaan cross-validation dan teknik tuning parameter lainnya.
- Gradient Boosting dapat cenderung overfit pada data training jika tidak dilakukan pengaturan parameter yang baik. 
- Gradient Boosting memerlukan jumlah data yang besar untuk memperoleh model yang akurat dan stabil. Jika jumlah data terlalu sedikit, algoritma ini dapat menjadi tidak stabil dan menghasilkan model yang tidak akurat.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## 6. Evaluation
Metrik evaluasi yang digunakan adalah sebagai berikut:

### Accuracy
*Accuracy* adalah metric yang mengukur seberapa sering model machine learning memprediksi keluaran hasil secara tepat. Hasil akurasi ini adalah pembagian prediksi yang benar dengan jumlah seluruh data.
Hasil akurasi ini berada diantara angka 0 dan 1. Semakin besar nilai akurasi, maka semakin baik model yang dibuat. Artinya, jika hasil accuracy bernilai 1, maka model mampu memprediksi seluruh data tanpa satupun kesalahan.
Untuk menghitung nilai akurasi, menggunakan rumus berikut:
- TP = True Positive (Model memprediksi dengan benar bahwa data bernilai positive)
- TN = True Negative (Model memprediksi dengan benar bahwa data bernilai negative)
- FP = False Positive (Model memprediksi dengan salah bahwa data bernilai positive)
- FN = False Negative (Model memprediksi dengan salah bahwa data bernilai negative)

### Recall
Recall/Sensitivitas adalah metrik evaluasi yang menggambarkan seberapa baik suatu model dalam mengidentifikasi positif dengan benar.
Sebagai analogi, bayangkan kita sedang mencari jarum di tumpukan jerami. Recall menggambarkan seberapa baik kita menemukan semua jarum yang ada di tumpukan tersebut. Jika kita menemukan 6 dari 10 dari jarum ditumpukan Jerami tersebut, artinya kita masih melewatkan 4 jarum yang belum ditemukan.
Dalam proyek ini, recall ini sangat penting, karena biaya salah mendiagnosis pasien positif (false negative) lebih tinggi. Contoh, pasien yang seharusnya menjalani pengobatan karena diabetes malah memakan makanan manis yang memperburuk penyakit diabetesnya.
Untuk menghitung recall, Anda dapat membagi jumlah true positive dengan jumlah contoh positif. Semakin tinggi recall, semakin banyak positif yang terdeteksi.

### Precision
Precision/Presisi adalah metrik yang mengukur seberapa sering model pembelajaran mesin memprediksi kelas positif dengan benar.
Presisi dihitung dengan membagi jumlah prediksi positif yang benar (positif benar) dengan jumlah total prediksi positif (positif benar ditambah positif salah).

### Confusion Matrix
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.
Untuk menghitung nilai akurasi, kita dapat menggunakan persamaan matematika berikut:



Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
