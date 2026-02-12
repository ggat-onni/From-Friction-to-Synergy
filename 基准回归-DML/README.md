# DML Model for Benchmark Regression

This project is a causal inference model library based on Double Machine Learning (DML). It is primarily designed for benchmark regression​ and heterogeneity analysis (subgroup regression)​ experiments in academic papers.
## 1. Environment Dependencies

This project is developed under **Python 3.7** .

### 1.1 System Requirements
* Python 3.7
* R language environment (required for rpy2)

### 1.2 Python Library Versions
To ensure reproducible experimental results, please strictly adhere to the following dependency versions:

* Numpy == 1.18.1
* pandas == 1.0.3
* Pytorch == 1.3.1
* scikit-learn == 0.22.1
* pathlib2 == 2.3.5
* scipy == 1.4.1
* matplotlib == 3.1.3
* pillow == 7.0.0
* rpy2 == 2.9.4

## 2. Data Preparation

The project runs on datasets in CSV format. Please ensure your data files meet the following format requirements:

| Variable Type | Column Name | 	Description                       |
| :--- | :--- |:-------------------------|
| **Dependent Variable** | `y` | 	Must be renamed to y                 |
| **Core Explanatory Variable** | `d` | Must be renamed to d                 |
| **Control Variables** | (Keep original names) | Variable names and symbols must match Table 1​ in the paper |

## 3. Usage Instructions

Since the data structure of CSV files varies across experiments, model hyperparameters (k value) need to be dynamically adjusted based on data dimensions. Please follow these steps precisely:

### Step 1: Set Data File
Open the empirical_application.pyfile in your code editor, locate the namevariable, and modify its value to the name of the CSV file for your current experiment.

### Step 2: Determine Model Parameters (k)
Since the **k** value is related to data dimensions, manually determine it when using new data for the first time:

1.Run empirical_application.pydirectly.

2.The program may throw an error indicating a dimension mismatch.

3.Record the number provided in the error message, which is the baseline kvalue.

4.Return to the code and modify the parameters for the following models according to the logic below:

* **model_nn1**: Set to the number from the error message
* **model_nn2**: Set to this number - 1
* **model_knn1**: Set to this number - 1
* **model_knn2**: Set to this number - 1

## 4. Experiment Configuration Quick Reference
To reproduce the experimental results from the paper, please refer to the table below to set parameters based on the experiment type.

### Variable Explanation
* **k (Main)**: Corresponds to the k parameter of model_nn1.
* **k (Others)**:  Corresponds to the k parameters of model_nn2, model_knn1, model_knn2.
* **t_list**: Corresponds to the time/threshold list setting in the code.

| Experiment Scenario | k (Main) | k (Others) | t_list Setting Code |
| :--- | :--- | :--- | :--- |
| **Baseline Model** | **303** | 302 | `np.arange(0, 3.75, 0.25)` |
| **Internal Heterogeneity** | **303** | 302 | `np.arange(0, 1.69, 0.2)` |
| **External Heterogeneity 1** | **296** (or 290) | 295 (or 289) | `np.arange(0, 3.75, 0.25)` |
| **External Heterogeneity 2** | **164** (or 196) | 163 (or 195) | `np.arange(0, 3.75, 0.25)` |

> **Note:​** The numbers in parentheses for external heterogeneity experiments (e.g., 290, 196) represent possible dimensions corresponding to different subgroup data. Always choose the correct value based on the error message as described in the "Usage Instructions".

## 5. Project Structure

```text
├── empirical_application.py   # Main program entry (modify name and k parameters here)
├── requirements.txt           # Dependency list
├── [Data Directory]           # Directory for CSV data files
└── README.md                  # Project documentation
