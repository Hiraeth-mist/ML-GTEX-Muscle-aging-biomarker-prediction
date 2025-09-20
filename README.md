# 🧬 ML-Powered Biological Age Prediction & Biomarker Discovery in Human Muscle 👵➡️👶

## Table of Contents
-   [🌟 Overview](#-overview)
-   [🤔 Motivation](#-motivation)
-   [📊 Data](#-data)
-   [🔬 Methodology](#-methodology)
    -   [Data Acquisition & Preprocessing](#data-acquisition--preprocessing)
    -   [Exploratory Data Analysis (EDA) & Visualization](#exploratory-data-analysis-eda--visualization)
    -   [Machine Learning Model Development](#machine-learning-model-development)
    -   [Hyperparameter Tuning](#hyperparameter-tuning)
    -   [Model Evaluation](#model-evaluation)
    -   [Biomarker Discovery & Gene Annotation](#biomarker-discovery--gene-annotation)
-   [🔑 Key Findings](#-key-findings)
    -   [Model Performance Summary](#model-performance-summary)
    -   [Top Muscle Aging Biomarkers](#top-muscle-aging-biomarkers)
-   [🛠️ Technologies Used](#️-technologies-used)
-   [🚀 How to Run This Project](#-how-to-run-this-project)
-   [🔮 Future Work & Extensions](#-future-work--extensions)
-   [📧 Contact Me](#-contact-me)

---

## 🌟 Overview

This project delves into the fascinating field of aging biology by leveraging **Machine Learning (ML)** to predict **biological age** from human **gene expression data**. Specifically, it focuses on **skeletal muscle tissue**, which undergoes significant changes with age (like sarcopenia). Through a robust bioinformatics and data science pipeline, this work aims to identify key **molecular biomarkers** – genes whose expression patterns are most indicative of an individual's age. This project showcases a comprehensive understanding of biological data handling, advanced machine learning techniques, and the critical step of translating computational findings into biological insights.

---

## 🤔 Motivation

Aging is a complex biological process leading to functional decline and increased susceptibility to various diseases. While chronological age is simply time passed, **biological age** reflects the true physiological state of an individual and can vary significantly. Understanding the molecular drivers of biological aging is crucial for developing therapeutic interventions and personalized medicine. Gene expression provides a dynamic snapshot of cellular activity, making it a rich source for uncovering these age-related molecular signatures.

This project specifically highlights my ability to:
* Navigate and analyze large-scale biological datasets.
* Apply cutting-edge AI/ML models to solve complex biomedical problems.
* Bridge wet lab biological understanding with computational analysis.
* Identify interpretable biomarkers with translational potential.

---

## 📊 Data

The core of this project utilizes publicly available data from the **Genotype-Tissue Expression (GTEx) Consortium (v8 and v10 releases)**.

* **Source:** [GTEx Portal](https://gtexportal.org/home/datasets)
* **Data Types:**
    * **Bulk RNA-sequencing Data:** Gene-level Transcripts Per Million (TPM) values from various human tissues (GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct).
    * **Subject Phenotypes:** Demographic and clinical information for human donors, including chronological age (GTEx_Analysis_v8_SubjectPhenotypes.tsv).
    * **Sample Attributes:** Detailed information about each collected sample, including its tissue type (GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv).
* **Target Tissue:** For focused analysis and computational efficiency, the project primarily analyzed data from **Skeletal Muscle** tissue.

---

## 🔬 Methodology

The project follows a rigorous multi-stage pipeline, from raw data to interpretable biological insights:

### Data Acquisition & Preprocessing
* Downloaded large GTEx expression (`.gct`) and metadata (`.tsv` / `.xlsx`) files.
* Implemented a custom, memory-optimized Python function (`read_gct_file_optimized_memory`) to selectively load only the necessary gene expression data, preventing memory crashes in cloud environments (like Google Colab's free tier).
* Parsed and transformed subject phenotype data, converting age bins (e.g., '60-69') into numerical midpoints (e.g., 64.5) for regression.
* Filtered and merged data to create a clean dataset specifically for **muscle tissue samples**, linking gene expression to donor age.
* Ensured data quality by checking for and handling missing values.

### Exploratory Data Analysis (EDA) & Visualization
* Visualized the distribution of age in the selected muscle samples to understand cohort representation.
* Plotted expression distributions of a few random genes to inspect data characteristics.

### Machine Learning Model Development
* **Data Splitting:** Divided the prepared data (`X`: gene expression features, `y`: age target) into 80% training and 20% testing sets.
* **Feature Scaling:** Standardized gene expression features using `StandardScaler` to ensure all genes contribute equally to model training, preventing high-expression genes from dominating.
* **Initial Model Training:** Trained a diverse set of baseline regression models:
    * `Linear Regression`
    * `Lasso Regression`
    * `Ridge Regression`
    * `Random Forest Regressor`
    * `Gradient Boosting Regressor`
* **Initial Model Evaluation:** Assessed the baseline performance of these models on the unseen test set using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared ($R^2$).

### Hyperparameter Tuning
* Focused on **XGBoost Regressor**, a highly optimized gradient boosting framework, for advanced tuning due to its superior performance on tabular data.
* Leveraged **GPU acceleration** in Google Colab (or Kaggle) for significantly faster training and tuning times.
* Employed **`RandomizedSearchCV`** with cross-validation to efficiently search for the optimal hyperparameters (`n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`), striking a balance between exploration and computational cost.

### Model Evaluation
* Evaluated the **best-tuned XGBoost model** on the unseen test set using MAE, RMSE, and $R^2$.
* Compared its performance against the initial baseline models to quantify the improvement gained from tuning and using a more advanced algorithm.

### Biomarker Discovery & Gene Annotation
* Extracted **feature importance scores** from the tuned XGBoost model (as well as Gradient Boosting and Lasso) to identify the top candidate genes influencing age prediction.
* Used the **`mygene` Python library** to programmatically convert Ensembl Gene IDs (e.g., `ENSG00000148468.16`) to human-readable **Gene Symbols**, **Gene Names**, and **Entrez IDs** for biological interpretability.
* Generated a **downloadable CSV table** of these annotated top biomarkers, including their importance scores and Lasso coefficients (indicating direction of association with age).

---

## 🔑 Key Findings

The project successfully developed a robust model for biological age prediction and identified a set of promising molecular biomarkers in human muscle.

### Model Performance Summary
| Model                           | MAE  (Years) | RMSE (Years) | R-squared ($R^2$) |
| :------------------------------ | :----------- | :----------- | :---------------- |
| Linear Regression               | 7.95         | 9.84         | 0.46              |
| Lasso Regression                | 7.21         | 8.96         | 0.55              |
| Ridge Regression                | 7.95         | 9.84         | 0.46              |
| Random Forest Regressor         | 8.19         | 9.84         | 0.46              |
| Gradient Boosting Regressor     | 7.21         | 8.61         | 0.59              |
| **Tuned XGBoost Regressor (GPU)** | **7.03** | **8.47** | **0.60** |

* The **Tuned XGBoost Regressor** is the best-performing model, achieving an MAE of **~7.03 years** and explaining **60% of the variance in age** in human muscle tissue ($R^2 = 0.60$). This demonstrates a strong and meaningful predictive capability for biological age from gene expression.

### Top Muscle Aging Biomarkers

The analysis yielded a prioritized list of genes most influential in predicting age. The `top_50_muscle_aging_biomarkers.csv` file provides the full annotated list.

**Conceptual Biological Interpretation:**
* Genes with positive correlations (from Lasso) could be considered "pro-aging" markers, while those with negative correlations might be "youth-associated" or "anti-aging" markers.
* These findings can be further validated in wet lab molecular biology for deeper interpretation using experimental validation strategies (e.g., confirming expression changes via qPCR, or protein levels via Western Blot in aged vs. young muscle samples).

---

## 🛠️ Technologies Used

* **Languages:** Python 🐍
* **Libraries:**
    * `pandas` (Data manipulation and analysis)
    * `numpy` (Numerical operations)
    * `matplotlib` (Data visualization)
    * `seaborn` (Statistical data visualization)
    * `scikit-learn` (Machine learning models, preprocessing, evaluation, tuning)
    * `xgboost` (GPU-accelerated Gradient Boosting)
    * `mygene` (Programmatic gene ID annotation)
    * `scipy` (Statistical functions for tuning)
* **Environment:** Google Colab / Kaggle Notebooks ☁️

---

## 🚀 How to Run This Project

1.  **Clone the Repository:**
    * git clone [https://github.com/Hiraeth-mist/ML-GTEX-Muscle-aging-biomarker-prediction.git](https://github.com/Hiraeth-mist/ML-GTEX-Muscle-aging-biomarker-prediction.git)
    * cd ML-GTEX-Muscle-aging-biomarker-prediction
2.  **Open in Google Colab:**
    * Go to [colab.research.google.com](https://colab.research.google.com/).
    * Click `File` -> `Upload notebook` -> `GitHub` tab. Paste this repository URL and select the `.ipynb` file.
3.  **Setup Google Drive (for Colab):**
    * The notebook expects GTEx data files in `/content/drive/My Drive/age_prediction/data/`.
    * Create this folder structure in your Google Drive and upload `GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct`, `GTEx_Analysis_v8_Annotations_SampleAttributesDS.tsv`, and `GTEx_Analysis_v8_SubjectPhenotypes.tsv` into it.
4.  **Set GPU Runtime:**
    * In Colab, go to `Runtime` -> `Change runtime type`.
    * Select `GPU` as the "Hardware accelerator" and click "Save".
5.  **Install Libraries:** The notebook includes `!pip install` commands for necessary libraries like `xgboost` and `mygene`. These will run automatically.
6.  **Run All Cells:** Go to `Runtime` -> `Run all`. The entire pipeline, from data loading to model tuning and biomarker table generation, will execute. The `top_50_muscle_aging_biomarkers.csv` file will be saved in your Google Drive.

---

## 🔮 Future Work & Extensions

* **Deeper Model Interpretation:** Implement SHAP (SHapley Additive exPlanations) values for a more granular understanding of individual gene contributions to age predictions.
* **Explore Other Advanced Models:** Experiment with LightGBM (another high-performance gradient boosting library) or even basic Neural Networks (`MLPRegressor`) if highly complex non-linear patterns are suspected.
* **Feature Selection & Dimensionality Reduction:** Apply techniques like PCA or feature selection algorithms to reduce the dimensionality of the gene expression data, potentially improving model robustness or interpretability with fewer genes.
* **Single-Cell RNA-seq Data:** Extend the analysis to single-cell RNA-seq data to identify age-related changes at the cell-type-specific level.
* **Integration with Other Omics Data:** Incorporate DNA methylation, proteomics, or metabolomics data (if available for GTEx samples) for a multi-omics perspective on aging.
* **Experimental Validation:** Conceptually outline strategies for validating the identified biomarkers in a wet lab setting (e.g., qPCR, Western Blot, immunohistochemistry in aged vs. young muscle samples).

---

## 📧 Contact Me

Feel free to reach out if you have any questions, suggestions, or collaboration opportunities!

**Email:** [akshat.bio@proton.me](mailto:akshat.bio@proton.me)
**LinkedIn:** [Akshat Jaiswal](https://www.linkedin.com/in/akshat-jaiswal-9234b5384/)
