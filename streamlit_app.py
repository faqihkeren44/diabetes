import pickle
import streamlit as st
import pandas as pd

# Membaca model
with open("model.pkl", "rb") as model_file:
    model_load = pickle.load(model_file)

# Tombol prediksi
kolom_data = ['name', 'age', 'blood_glucose_level', 'HbA1c_level', 'bmi']

with st.sidebar:
    st.header('Predict Machine', divider='rainbow')

    st.subheader('Gender')
    name = st.text_input('Your Name')
    age = st.number_input('Insert your age', 0, 80)
    blood_glucose_level = st.number_input('Insert Glucose Level', 0, 500)
    HbA1c_level = st.number_input('Insert your Hemoglobin Level', 0, 100)
    bmi = st.slider('Insert your bmi', 0, 200)
 
    new_data = [[name, age, blood_glucose_level, HbA1c_level, bmi]]
    
    if st.sidebar.button('Predict'):
        new_data = pd.DataFrame(new_data, columns=kolom_data)
        result = model_load.predict(new_data)
        if(result[0] == 0):
            st.success('Siswa tidak dropout')
            st.balloons()
        else:
            st.error('Siswa dropout')

# Judul web
st.title('Jaya Jaya Institut')
st.subheader('Apa itu Jaya Jaya Institut?',  divider='grey')
st.markdown('''Jaya Jaya Institut merupakan pendidikan perguruan internasional yang
            sudah menarik perhatian berbagai kalangan siswa, bahkan dari mancanegara.
            Institut ini sudah berdiri sejak tahun 2000. Dengan tawaran beasiswa yang''')
st.subheader('Apa fungsi Predict Machine di samping?',  divider='grey')
st.markdown('''Predict Machnine adalah sebuah mesin model yang dibuat untuk kebutuhan
            prediksi. Mesin ini dibuat dan dilatih berdasarkan data yang ada pada
            Jaya Jaya Institute.''')
st.subheader('Bagaimana cara menggunakan Predict Machine?',  divider='grey')
st.markdown('''Untuk menampilkan mesin model klik tombol panah (>) di pojok kiri atas
            halaman. Pada tampilannya, ada beberapa inputan yang harus diisi, yaitu:''')
st.text('Gender: Jenis kelamin siswa (Man = Pria, Woman = Wanita)')
st.text('Marital Status: Status pernikahan (Single = Belum menikah, Married = Menikah, Widower = Janda/Duda, Divorced = Cerai, Facto Union = Pernikahan tidak resmi, Legally Separated = Perceraian resmi)')
st.text('Age At Enrollment: Usia siswa saat melakukan enrollment')
st.text('Debtor: Apakah siswa memiliki tanggungan hutang?')
st.text('Tuition Fees: Apakah siswa sudah melunasi pembayaran terkini?')
st.text('Scholarship Holder: Apakah siswa mendapatkan beasiswa?')
st.text('Displaced: Apakah siswa berasal dari keluarga kurang mampu?')
st.text('Curriculum Unit: Jumlah Curriculum yang dikreditkan, dienrollment, disetujui, dan nilainya. Baik pada semester 1 maupun semester 2')