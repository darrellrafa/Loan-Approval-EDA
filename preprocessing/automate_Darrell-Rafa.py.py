import pandas as pd
import argparse
import os

def load_data(file_path):
    """Fungsi untuk memuat dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def clean_and_preprocess(df):
    """Fungsi untuk membersihkan dan memproses data"""
    print("Memulai proses pembersihan data...")
    
    # 1. Membersihkan spasi pada nama kolom
    df.columns = df.columns.str.strip()
    
    # 2. Menghapus kolom yang tidak relevan
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        
    # 3. Membersihkan spasi pada isi data teks
    cat_cols = ['education', 'self_employed', 'loan_status']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip()
            
    # 4. Feature Engineering: Menggabungkan total aset
    df['total_assets_value'] = (df['residential_assets_value'] + 
                                df['commercial_assets_value'] + 
                                df['luxury_assets_value'] + 
                                df['bank_asset_value'])
    
    # Menghapus kolom aset individu agar tidak redundan
    df = df.drop(columns=['residential_assets_value', 
                          'commercial_assets_value', 
                          'luxury_assets_value', 
                          'bank_asset_value'])
    
    # 5. Encoding Label Target (Approved=1, Rejected=0)
    df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # 6. One-Hot Encoding untuk fitur kategorikal lainnya
    kategorikal_kolom = ['education', 'self_employed']
    df_cleaned = pd.get_dummies(df, columns=kategorikal_kolom, drop_first=True)
    
    print("Preprocessing selesai!")
    return df_cleaned

def save_data(df, output_path):
    """Fungsi untuk menyimpan data bersih"""
    df.to_csv(output_path, index=False)
    print(f"Data bersih berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    # Menggunakan argparse agar script bisa dijalankan lewat terminal dengan parameter
    parser = argparse.ArgumentParser(description="Otomatisasi Preprocessing Dataset Loan Approval")
    parser.add_argument('--input', type=str, required=True, help="Path ke file dataset mentah")
    parser.add_argument('--output', type=str, required=True, help="Path untuk menyimpan dataset bersih")
    
    args = parser.parse_args()
    
    # Menjalankan alur pipeline
    try:
        raw_data = load_data(args.input)
        cleaned_data = clean_and_preprocess(raw_data)
        save_data(cleaned_data, args.output)
    except Exception as e:
        print(f"Terjadi kesalahan saat memproses data: {e}")