# Steam-Review-Sentiment-Analysis
Proyek Analisis Sentimen Ulasan Game di Steam menggunakan algoritma Naive Bayes dan SVM (LinearSVC).
Analisis Sentimen Ulasan Game di Platform Steam Menggunakan Naive Bayes Classifier dan SVM dengan TF-IDF
Anggota Kelompok:

Rizqi Agung Raya (NIM: 2330511037)

Rian Alpriyanda Kurniawan (NIM: 2330511025)

Project Overview
<img width="640" height="832" alt="steam-reviews-useful-as-always-v0-99jtzxwn73dd1" src="https://github.com/user-attachments/assets/84af3329-5dfd-4455-9b78-d368a1228c22" />

Industri permainan video menerima ribuan hingga jutaan ulasan setiap harinya di platform distribusi digital seperti Steam. Banyaknya jumlah ulasan berupa teks bebas (free-text) merupakan permasalahan tersendiri bagi para pengembang (developer) game. Membaca ulasan secara manual untuk mengetahui apakah sentimen pemain positif atau negatif sangat tidak efisien dan memakan waktu.

Oleh karena itu, otomasi analisis sentimen memiliki peran penting bagi developer untuk dengan cepat menangkap feedback pemain, yang pada akhirnya dapat memengaruhi pembaruan game (patch) dan kesuksesan komersial perusahaan. Proyek ini bertujuan untuk membangun model Machine Learning yang dapat mengklasifikasikan ulasan pemain menjadi sentimen Positif (Recommended) atau Negatif (Not Recommended) menggunakan pendekatan pemrosesan bahasa alami (NLP) dengan membandingkan dua algoritma: Naive Bayes dan Support Vector Machine (SVM).

💡 Manfaat Proyek:
✔ Membantu developer merangkum jutaan feedback teks menjadi metrik sentimen yang terukur.
✔ Mengotomatisasi proses penyortiran ulasan positif dan negatif.
✔ Memberikan wawasan (insight) secara real-time mengenai kepuasan pemain terhadap suatu game.

Business Understanding
📝 Problem Statements

Bagaimana cara mengekstrak informasi bermakna dari teks ulasan pemain yang tidak terstruktur?

Seberapa efektif metode Naive Bayes Classifier dibandingkan dengan Support Vector Machine (SVM) dalam memprediksi sentimen ulasan game?

🎯 Goals

Mengembangkan sistem yang dapat membersihkan dan memproses teks ulasan Steam berbahasa Inggris.

Membandingkan performa model klasifikasi sentimen antara Naive Bayes dan SVM untuk memahami pendekatan terbaik.

Mendeploy model terbaik ke dalam antarmuka berbasis web agar mudah digunakan oleh pengguna akhir.

🛠 Solution Approach

Ekstraksi Fitur (TF-IDF): Menggunakan Term Frequency-Inverse Document Frequency untuk mengubah teks ulasan menjadi representasi vektor numerik.

Modeling Komparatif: Melatih data menggunakan dua algoritma (Multinomial Naive Bayes dan LinearSVC) untuk menemukan model dengan akurasi tertinggi.

Data Understanding
Dataset yang digunakan diperoleh dari platform Kaggle: Sentiment Analysis for Steam Reviews. Pada tahap ini, kita memahami struktur dataset yang berfokus pada kolom teks ulasan dan label rekomendasi.

📂 Dataset Components:
Dataset utama (train.csv) memiliki ribuan baris, namun proyek ini berfokus pada dua kolom esensial untuk NLP:

📌 Uraian Fitur Utama:

user_review: Berisi teks ulasan lengkap dari pemain mengenai pengalaman mereka bermain game di Steam (Tipe: String).

user_suggestion: Label target klasifikasi, di mana bernilai 1 jika pengguna merekomendasikan game (Positif), dan 0 jika tidak merekomendasikan (Negatif) (Tipe: Integer).
<img width="1091" height="416" alt="3fccfc6e-2b91-44f5-861b-f3b04d426506" src="https://github.com/user-attachments/assets/774ecb3c-560a-41d3-bc83-669fdd62235f" />

🔍 Eksplorasi Data (EDA)
Dari hasil eksplorasi data, ditemukan bahwa:

Distribusi Kelas: Dataset memiliki dua kelas target, yaitu kelas 1 (Rekomendasi) sebanyak 9.968 ulasan dan kelas 0 (Tidak Rekomendasi) sebanyak 7.526 ulasan.
<img width="762" height="512" alt="6b38a9e3-d8f7-4d8c-bf79-a749e28ca2c4" src="https://github.com/user-attachments/assets/714630cb-2473-4084-babc-a3d249b3828f" />

Karakteristik Teks: Teks ulasan mengandung banyak noise seperti tanda baca, angka, karakter khusus, dan huruf kapital/kecil yang tidak seragam, sehingga memerlukan pembersihan secara sistematis.

Data Preparation
Tahapan data preparation dilakukan secara bertahap untuk memastikan teks ulasan siap dikonsumsi oleh algoritma Machine Learning.

📌 Pembersihan Teks (Text Cleaning)
Langkah pertama adalah membersihkan teks pada kolom user_review dengan cara:
✔ Case Folding: Mengubah seluruh teks menjadi huruf kecil (lowercase).
✔ Punctuation Removal: Menghapus angka dan karakter khusus menggunakan Regular Expression (Regex).
✔ Stopword Removal: Menghapus kata-kata umum yang tidak memiliki makna signifikan (seperti "the", "is", "in") menggunakan library NLTK berbahasa Inggris.

📌 Pembagian Data (Train-Test-Split)
Dataset teks yang sudah bersih dibagi menjadi data latih (training data) dan data uji (testing data) dengan proporsi 80:20 agar hasil eksperimen dapat divalidasi.

📌 Ekstraksi Fitur (TF-IDF Vectorization)

Teks diubah menjadi representasi numerik menggunakan TfidfVectorizer.

Fitur dibatasi menjadi 5000 kata paling penting (max_features=5000) untuk mengurangi beban komputasi dan menghindari noise dari kata-kata yang jarang muncul.

Modeling and Results
📝 Pendekatan Model

Pada proyek ini, kami membandingkan dua algoritma klasifikasi teks yang populer:

Naive Bayes Classifier (MultinomialNB):
Algoritma probabilistik yang didasarkan pada Teorema Bayes. Algoritma ini sangat populer untuk analisis teks karena asumsi independensi antar fiturnya membuatnya sangat cepat dilatih.

Support Vector Machine (LinearSVC):
Algoritma yang bekerja dengan mencari hyperplane (garis pembatas) terbaik yang memaksimalkan margin antara kelas sentimen positif dan negatif. LinearSVC dirancang khusus untuk menangani dataset teks dengan dimensi tinggi secara efisien.

✔ Tahapan Model:

Inisialisasi kedua model.

Melatih masing-masing model menggunakan matriks TF-IDF dari data latih.

Melakukan prediksi pada data uji untuk membandingkan performanya.

Evaluation Model
Evaluasi model klasifikasi dilakukan dengan membandingkan hasil prediksi model terhadap data uji yang nilai sebenarnya sudah diketahui menggunakan metrik Accuracy, Precision, Recall, dan F1-Score.

📌 Hasil Evaluasi:
<img width="702" height="241" alt="731bc580-6b11-4f38-bf58-a205922cade7" src="https://github.com/user-attachments/assets/8ef698fc-1ec8-4153-ae50-bca5eb175edd" />
Naive Bayes Classifier: Secara umum memberikan hasil yang cukup baik dan eksekusi yang sangat cepat, namun performanya sedikit di bawah SVM dalam membedakan konteks kalimat yang kompleks.

Support Vector Machine (SVM): Menunjukkan akurasi tertinggi di atas 85%. Model SVM memperlihatkan skor precision dan recall yang sangat seimbang (0.86 dan 0.88 untuk kelas 1), menandakan bahwa model ini tidak bias dan sangat robust.

Kesimpulan Evaluasi:
✔ Algoritma LinearSVC (SVM) yang dipadukan dengan TF-IDF terbukti lebih mumpuni dibandingkan Naive Bayes dalam membedakan sentimen teks ulasan Steam pada dataset ini. Oleh karena itu, SVM dipilih sebagai model utama yang akan digunakan untuk tahap Deployment.

Deployment (⭐ Poin Plus)
Selain melatih model, kami juga mendeploy model SVM (LinearSVC) terbaik kami ke lingkungan produksi sederhana agar dapat diuji secara interaktif.

Model yang telah dilatih diekspor menjadi file .pkl menggunakan library joblib dan di-deploy dalam bentuk aplikasi web menggunakan antarmuka Gradio yang di-hosting secara publik di Hugging Face Spaces. Metode ini dipilih karena memberikan pengalaman pengguna (User Experience) yang jauh lebih baik dan user-friendly dibandingkan hanya menggunakan script di terminal.

🔗 Tautan Aplikasi Live: [https://rizqi19-analisis-sentimen-game-steam.hf.space/?__theme=system&deep_link=LFaZvB9JDng]

Aplikasi ini memungkinkan developer maupun pengguna umum untuk mengetikkan ulasan game berbahasa Inggris secara real-time, dan sistem akan langsung memprediksi apakah ulasan tersebut bernada rekomendasi (Positif) atau tidak (Negatif).

Kesimpulan
🌟 Pembersihan data teks (Text Cleaning) yang komprehensif sangat krusial dalam meningkatkan kualitas fitur sebelum dimasukkan ke dalam model klasifikasi.
🌟 Dalam perbandingan model, SVM (LinearSVC) terbukti sedikit lebih superior dibandingkan Naive Bayes Classifier untuk klasifikasi teks berdimensi tinggi menggunakan pembobotan TF-IDF.
🌟 Melalui deployment di antarmuka Gradio (Hugging Face), proyek Machine Learning tidak hanya berakhir sebagai metrik evaluasi di notebook, melainkan menjadi sebuah produk yang siap digunakan untuk menyelesaikan permasalahan otomasi sentimen yang dirumuskan pada tahap Business Understanding.
