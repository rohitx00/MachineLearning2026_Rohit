# Placement Prediction Project

## Overview

This project implements a machine learning model to predict student placement outcomes based on various academic and co-curricular attributes. The model uses **Logistic Regression** to classify whether a student will be placed or not.

## Project Structure

### 📁 Folder Contents

#### 1. **placement.ipynb**

- Main Jupyter notebook containing the complete ML pipeline
- **Contents:**
  - Data loading and exploration
  - Data preprocessing (handling categorical variables, feature encoding)
  - Feature engineering and data normalization
  - Train-test split implementation
  - Model training using Logistic Regression
  - Model evaluation with accuracy metrics
  - Decision boundary visualization
- **Key Libraries Used:**
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `scikit-learn`: Machine learning algorithms and preprocessing
  - `mlxtend`: Visualization utilities for decision regions

#### 2. **dataset/college_placement.csv**

- Original dataset containing student placement records
- **Features:**
  - `College_ID`: Unique identifier for each student
  - `IQ`: Intelligence quotient score
  - `Prev_Sem_Result`: Previous semester result/GPA
  - `CGPA`: Cumulative Grade Point Average
  - `Academic_Performance`: Overall academic performance score
  - `Internship_Experience`: Whether student has internship experience (Yes/No)
  - `Extra_Curricular_Score`: Score for extra-curricular activities
  - `Communication_Skills`: Communication skills rating
  - `Projects_Completed`: Number of projects completed
  - `Placement`: Target variable - placement status (Yes/No)
- **Data Format:** CSV with headers
- **Usage:** Training and testing the ML model

#### 3. **placement_prediction_model.pkl**

- Serialized trained machine learning model (Logistic Regression)
- **Purpose:**
  - Stores the fitted model for future predictions
  - Allows model reuse without retraining
  - Can be loaded for inference on new data
- **Format:** Python pickle file

#### 4. **README.md**

- Documentation file (this file)
- Provides project overview and usage instructions

## Model Details

### Algorithm: Logistic Regression

- **Type:** Binary Classification
- **Target Variable:** Placement (0 = Not Placed, 1 = Placed)
- **Preprocessing:**
  - StandardScaler normalization applied to all features
  - Categorical variables encoded as binary (0/1)
  - Outliers handled through standardization

### Model Performance

- **Metric:** Accuracy Score
- **Evaluation Set:** 20% test data (80-20 split)
- **Random State:** 42 (for reproducibility)

## Data Preprocessing Steps

1. **Data Loading:** Read CSV file using pandas
2. **Feature Encoding:**
   - `Placement`: 'No' → 0, 'Yes' → 1
   - `Internship_Experience`: 'No' → 0, 'Yes' → 1
3. **Feature Renaming:** Column names converted to lowercase for consistency
4. **Feature Dropping:** `College_ID` removed (not useful for prediction)
5. **Scaling:** StandardScaler applied to normalize feature ranges

## How to Use

### 1. **Run the Notebook**

```python
# Open placement.ipynb in Jupyter
# Execute cells sequentially to train the model
```

### 2. **Make Predictions**

```python
# Load the trained model
import pickle
with open('placement_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions on new data
prediction = model.predict(scaled_features)
```

### 3. **Load and Explore Dataset**

```python
import pandas as pd
df = pd.read_csv('dataset/college_placement.csv')
df.head()  # View first 5 rows
df.info()  # Get dataset information
df.describe()  # Get statistical summary
```

## Results & Insights

- Model provides binary predictions (Placed/Not Placed)
- Decision boundaries visualized for first two principal features
- Accuracy metrics help evaluate model performance
- Feature scaling ensures equal importance in the prediction

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- mlxtend

## Future Improvements

- Experiment with other classification algorithms (Random Forest, SVM, etc.)
- Perform feature importance analysis
- Apply dimensionality reduction (PCA)
- Implement cross-validation for better generalization
- Handle class imbalance if present
- Hyperparameter tuning for model optimization
