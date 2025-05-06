# Employee Attrition Analysis – Jaya Jaya Maju

**Submission by Malikusfz for Dicoding**

---

## 1. Business Understanding

**Problem Statement:**  
Jaya Jaya Maju (JJM) is experiencing an annual attrition rate exceeding **10%**, surpassing the industry average of approximately **6%**. High turnover increases recruitment costs, disrupts project continuity, and erodes the organizational knowledge base.

**Objectives:**  
The HR department has commissioned an analysis to:

1. Identify key drivers of employee attrition.
2. Develop an interactive dashboard for real-time monitoring of attrition drivers.
3. Build a predictive model to flag employees at high risk of leaving.

---

## 2. Data Understanding & Preparation

**Dataset Overview:**  
The dataset contains employee records with features such as `Age`, `Department`, `Tenure`, `Overtime`, and `Attrition` (target variable). `EmployeeId` serves as the unique identifier.

**Preprocessing Pipeline:**

1. **Data Cleaning:**
   - Removed 412 records with missing `Attrition` values.
   - Validated `EmployeeId` uniqueness to ensure data integrity.
2. **Feature Engineering:**
   - Applied one-hot encoding to 9 categorical variables (e.g., `Department`, `JobRole`).
   - Retained original numerical features (e.g., `Age`, `Tenure`) for model training.
3. **Dataset Partitioning:**
   - Used stratified sampling to split data: 80% training, 20% test, preserving class distribution.

**Documentation:**  
Detailed preprocessing steps are documented in the accompanying Jupyter notebook (`preprocessing.ipynb`).

---

## 3. Exploratory Data Analysis (EDA)

**Key Insights:**

- **Departmental Trends:** The **Sales** department has the highest attrition rate at **20%**, followed by **Human Resources** at **15%**.
- **Overtime Impact:** Employees frequently working overtime are **2.7×** more likely to resign.
- **Demographic Patterns:** Employees aged **<30 years** with **≤2 years** of tenure are the most prone to attrition.

**Visualization:**  
An interactive dashboard visualizing attrition trends is available:  
![Attrition Dashboard](assets/dashboard.png)

---

## 4. Modeling

**Model Selection & Performance:**  
Multiple models were evaluated, with hyperparameter tuning performed using Optuna for the best-performing model.

| Model                           | Accuracy | ROC-AUC  |
| ------------------------------- | -------- | -------- |
| Baseline (Majority Class)       | 59%      | 0.50     |
| **CatBoost (Optuna-optimized)** | **85%**  | **0.89** |

**Production Assets:**

- **Trained Model:** `attrition_model.pkl`
- **Inference API:** `prediction.py`

**Inference Example:**

```bash
python prediction.py '{"Age": 35, "Department": "Sales", "Tenure": 3, "Overtime": "Yes"}'
```

**Deployment Notes:**  
The model is deployed as a REST API for real-time predictions, integrated with the HR dashboard for seamless monitoring.

---

## 5. Conclusion & Recommendations

**Summary:**  
The analysis identified critical attrition drivers, including departmental differences, overtime, and demographic factors. The CatBoost model provides robust predictions, and the dashboard enables proactive HR interventions.

**Recommendations:**

1. Implement targeted retention strategies for high-risk groups (e.g., young employees in Sales).
2. Review overtime policies to reduce burnout.
3. Regularly update the model with new employee data to maintain predictive accuracy.

**Next Steps:**

- Deploy the dashboard to HR’s internal systems.
- Schedule quarterly model retraining to adapt to evolving trends.

---

**Artifacts:**

- Jupyter Notebook: `analysis_notebook.ipynb`
- Model File: `attrition_model.pkl`
- Dashboard: `dashboard.html`
- Inference Script: `prediction.py`
