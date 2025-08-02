graph TD
    subgraph "Tahap 1: Akuisisi Data"
        A[Mulai] --> B{Scrape Daftar Link};
        B --> C[Simpan Link ke CSV];
        C --> D{Scrape Detail Properti};
        D --> E[Buat Database Mentah (.csv)];
    end

    subgraph "Tahap 2: Persiapan & Pemodelan (di Colab)"
        E --> F{Muat Database Mentah};
        F --> G[Preprocessing & Cleaning Data];
        G -- "Handle NaN, Hapus Outlier, Feature Engineering" --> H{Bagi Data (Fitur X & Target y)};
        H --> I[Buat Preprocessing Pipeline];
        I -- "StandardScaler & OneHotEncoder" --> J{Latih & Evaluasi Beberapa Model};
        J -- "LightGBM, CatBoost, dll." --> K{Pilih Model Terbaik};
        K --> L[Latih Ulang Model Terbaik dengan Seluruh Data];
        L --> M[Ekstrak/Simpan Model Final (.joblib)];
    end

    subgraph "Tahap 3: Deployment"
        M --> N{Siapkan File Aplikasi};
        N -- "app.py, model.joblib, data.csv, requirements.txt" --> O[Upload ke GitHub];
        O --> P{Deploy di Streamlit Cloud};
        P --> Q[Aplikasi Live & Selesai];
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#f9f,stroke:#333,stroke-width:2px
