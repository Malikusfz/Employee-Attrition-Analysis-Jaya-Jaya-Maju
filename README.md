# Employee Attrition Analysis – Jaya Jaya Maju

_Dicoding Submission – **Malikusfz**_

---

## 1  Business Understanding

Jaya Jaya Maju (JJM) mengalami angka attrition tahunan **> 10 %**, melebihi rata‑rata industri ± 6 %.  
Turnover tinggi menaikkan biaya rekrutmen, mengganggu kontinuitas proyek, dan mengikis knowledge base.  
HR meminta kami untuk:

1. **Mengidentifikasi driver attrition.**
2. **Membangun dashboard** agar HR bisa memonitor driver tersebut secara real‑time.
3. **Melatih model prediksi** untuk menandai karyawan berisiko tinggi keluar.

---

## 2  Data Understanding & Preparation

| File                | Baris | Kolom | Target                            |
| ------------------- | ----- | ----- | --------------------------------- |
| `employee_data.csv` | 1 470 | 35    | `Attrition` (1 = leave, 0 = stay) |

Langkah utama:

1. **Cleaning** – Menghapus 412 baris dengan `Attrition` NA; memastikan `EmployeeId` unik.
2. **Feature engineering** – One‑hot encode 9 kolom kategorikal; fitur numerik dipertahankan.
3. **Split** – Train 80 %, test 20 % (stratified).

Detail lengkap di notebook.

---

## 3  Exploratory Data Analysis

- Departemen **Sales** mencatat attrition tertinggi (20 %) diikuti HR (15 %).
- **Overtime reguler** membuat risiko resign 2,7 × lebih tinggi.
- Karyawan muda (< 30 th) & masa kerja ≤ 2 th paling rentan.

![Dashboard teaser](Malikusfz-dashboard.png)

---

## 4  Modelling

| Model                                   | Accuracy | ROC‑AUC  |
| --------------------------------------- | -------- | -------- |
| Majority class                          | 59 %     | 0.50     |
| **CatBoost (Optuna tuned, 1500 iters)** | **85 %** | **0.89** |

Artefak model: **`attrition_model.pkl`**  
Wrapper CLI: **`prediction.py`**

```bash
python prediction.py '{"Age":35,"Department":"Sales", ...}'
```
