# **DEEP TABULAR LEARNING TO ANALYZE DATA SETS**

## Codigos de Colab:

# **Keras Tuner**

- [Keras Tuner](https://colab.research.google.com/drive/1m29_ro5Gec99015wEWBY0gN9ipNmczRh?usp=sharing)

**Resultados:**

| Model      | Accuracy | Precision | Recall  | F1-score | AUC      |
| ---------- | -------- | --------- | ------- | -------- | -------- |
| Best Model | 0.8625   | 0.905405  | 0.79056 | 0.844094 | 0.907207 |

---

# **Keras Tuner 2**

- [Keras Tuner 2](https://colab.research.google.com/drive/1j_NwKaNmTsc4lmoYBSj9_f5-9jaOfLKI?usp=sharing)

**Resultados:**

| Model      | Accuracy | Precision | Recall   | F1-score | AUC      |
| ---------- | -------- | --------- | -------- | -------- | -------- |
| Best Model | 0.66311  | 0.61165   | 0.258197 | 0.363112 | 0.620374 |

---

# **Random Forest**

- [Random Forest](https://colab.research.google.com/drive/1Sur69pbRfaW_rZariXwnLHj1J3_ijNrx?usp=sharing)

**Resultados**

|              | Precision | Recall | F1-score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Class 0      | 0.84      | 0.86   | 0.85     | 354     |
| Class 1      | 0.83      | 0.81   | 0.82     | 315     |
| Accuracy     |           |        | 0.84     | 669     |
| Macro avg    | 0.84      | 0.83   | 0.83     | 669     |
| Weighted avg | 0.84      | 0.84   | 0.84     | 669     |

|         | Predicted Class 0 | Predicted Class 1 |
| ------- | ----------------- | ----------------- |
| Class 0 | 303               | 51                |
| Class 1 | 59                | 256               |

---

# **Random Forest Tuning**

- [**Random Forest Tuning**](https://colab.research.google.com/drive/1tmehgHV5ausGBqixeyBU0mCdNemELaMT?usp=sharing)

**Resultados**

| Model                | Train Accuracy % | Test Accuracy % | Train Log Loss | Test Log Loss | Train ROC AUC | Test ROC AUC |
| -------------------- | ---------------- | --------------- | -------------- | ------------- | ------------- | ------------ |
| K-Nearest Neighbor   | 80.65            | 71.51           | 6.974179e+00   | 10.269699     | 0.81          | 0.72         |
| Logistic Regression  | 53.20            | 48.41           | 1.686778e+01   | 18.595135     | 0.53          | 0.49         |
| Random Forest        | 100.00           | 83.96           | 2.220446e-16   | 5.782937      | 1.00          | 0.84         |
| Gaussian Naive Bayes | 55.97            | 54.22           | 1.586969e+01   | 16.501313     | 0.56          | 0.54         |
| Random Forest Tuned  | 99.27            | 82.71           | 2.619996e-01   | 6.231614      | 0.99          | 0.83         |
