import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os


# input-output file
raw_file = r"C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\StudentsPerformance_raw\StudentsPerformance.csv"
output_folder = r"C:\Rahmi\kuliah\SEMESTER 7\ASAH\Eksperimen_SML_Rahmi-Dwi-Intan\Preprocessing\StudentsPerformance_preprocessing"
output_file = os.path.join(output_folder, "dataset_preprocessed.csv")


# Buat folder jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Baca dataset
data = pd.read_csv(raw_file)
print("Dataset berhasil dibaca:")
print(data.head())


# kolom numerik dan kategorikal
num_cols = ['writing score', 'math score', 'reading score']
cat_cols = ['parental level of education ', 'test preparation course', ' race/ethnicity', 'lunch', 'gender']

# preprocessing
scaler = StandardScaler()
encoder = OneHotEncoder(sparse_output=False, drop='first')

transformer = ColumnTransformer([
    ('num', scaler, num_cols),
    ('cat', encoder, cat_cols)
])

processed_data = transformer.fit_transform(data)

# nama kolom hasil encoding
encoded_features = transformer.named_transformers_['cat'].get_feature_names_out(cat_cols)

final_columns = num_cols + list(encoded_features)

# simpan ke DataFrame biar rapi
df_result = pd.DataFrame(processed_data, columns=final_columns)

# simpan hasil preprocessing
df_result.to_csv(output_file, index=False)

print(f"Hasil preprocessing disimpan di: {output_file}")