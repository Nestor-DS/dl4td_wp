# **DEEP TABULAR LEARNING TO ANALYZE DATA SETS**

# **deep_tabular-drinking_water_potability-**
Application of different deep tabular learning methods to water-related problem datasets using Python in order to find better prediction results.

# Deep Tabular Learning Project for Water Quality

This project focuses on the application of deep learning methods to related tabular datasets in order to find the best possible results. Different approaches are used for data exploration, classification with random forests and deep learning techniques.
It is intended to find an application to SuperTML - Sun et al., 2019 using data from: "___" . 

# Project Structure

## dataExploration
Modules and notebooks related to data exploration.

- `exploration/data_exploration.py`: Script for data exploration.
- `exploration/I-I(data_exploration).ipynb`: Detailed exploration of the dataset.

### Notebooks:
- `I-I(data_exploration).ipynb`: Detailed exploration of the dataset.
- `II-I(RandomForestClassifier).ipynb`: Implementation and evaluation of Random Forests.

## models
Notebooks related to model building and evaluation.

- `models/dl4td_keras_tuner_1.ipynb`: Notebook for Keras Tuner model development.
- `models/dl4td_keras_tuner_2.ipynb`: Another notebook for Keras Tuner model development.
- `models/dl4td_LDRSKN.ipynb`: Notebook for LDRSKN model development.
- `models/dl4td_random_forest_1.ipynb`: Notebook for Random Forest model development.
- `models/dl4td_random_forest_2.ipynb`: Another notebook for Random Forest model development.
- `models/dl4td-water-M1.ipynb`: Notebook for water model development.
- `models/MLP_1.ipynb`: Notebook for Multilayer Perceptron model development.
- `models/MLP_Tuner.ipynb`: Notebook for MLP model tuning.
- `models/modelsTest.ipynb`: Notebook for testing different models.
- `models/XGBoost.ipynb`: Notebook for XGBoost model development.

## SuperTML
Modules and notebooks related to SuperTML.

- `SuperTML/data_preparation.py`: Script for data preparation.
- `SuperTML/image_guardada.png`: Image saved for reference.
- `SuperTML/model.py`: Implementation of SuperTML model.
- `SuperTML/paper.png`: Image of paper related to SuperTML.
- `SuperTML/result.ipynb`: Notebook for analyzing results.
- `SuperTML/SuperTML.md`: Markdown file containing details about SuperTML.
- `SuperTML/train.py`: Script for training SuperTML model.

## Miscellaneous
Other files and scripts.

- `dl4td-water-1.ipynb`: Notebook for water analysis.
- `impute_data.py`: Script for data imputation.
- `README.md`: You are here! Overview of the project structure.

## Data Files
Datasets used in the project.

- `drinking_water_potability.csv`: Dataset on water potability.



# Usage

The model implementation and scanning initially use a clean dataset, but if your dataset is not clean you can use the `dl4td-water-1.ipynb` or the `impute_data.py`.

# Dependencies

Make sure to have the following Python libraries installed:

## Libraries used

## Data manipulation and visualization

- pandas
- seaborn
- matplotlib.pyplot

## Data preprocessing and modeling

- pandas
- sklearn
- tensorflow.keras

## Model evaluation

- sklearn.metrics

## Others

- joblib
- numpy
- PIL
- torch
- torchvision
- dmba.plotDecisionTree
- tkinter
- widgets
- IPython.display

# Visualization and Graphics

- matplotlib.pyplot
- seaborn
- cv2
- mpl_toolkits.mplot3d


## Results

# **Summary of Model Implementations:**

**Random Forest Tuning:**

* Precision: 99.1% (for class 0)
* Recall: 72.35% (for class 1)
* F1 Score: 72.35%
* Accuracy: 72.35%

The results demonstrate high precision for class 0 and reasonable recall for class 1. Additionally, the overall accuracy of the model is 72.35%, indicating good overall performance. Furthermore, the results remain stable after hyperparameter tuning.

**Random Forest:**

* Precision: 100% (for class 0)
* Recall: 74.44% (for class 1)
* F1 Score: 74.44%
* Accuracy: 74.44%

While this model exhibits perfect precision for class 0, the recall for class 1 is slightly lower than that of the tuned model, suggesting slightly inferior performance.

**Tune MLP 3:**

* Precision: 64% (for class 0)
* Recall: 63% (for class 1)
* F1 Score: 63%
* Accuracy: 63%

This model has reasonable precision and recall, but its overall performance is lower compared to the Random Forest models.

**Tune MLP 2:**

* Precision: 64% (for class 0)
* Recall: 63% (for class 1)
* F1 Score: 63%
* Accuracy: 63%

This model yields similar results to Tune MLP 3 but ranks slightly lower due to order.

**Tune MLP 1:**

* Precision: 52% (for class 0)
* Recall: 52% (for class 1)
* F1 Score: 52%
* Accuracy: 52%

This model has the lowest precision, recall, and F1 score, indicating inferior performance compared to the other models.

Based on these results, the Random Forest Tuning model is the best choice as it exhibits high precision for class 0, reasonable recall for class 1, and good overall accuracy. Additionally, the results remain consistent after hyperparameter tuning.