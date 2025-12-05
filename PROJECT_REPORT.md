# Breast Cancer Classification Using Machine Learning with Genetic Algorithm Feature Selection

## Project Report

**Date:** December 5, 2025  
**Domain:** Healthcare / Medical AI  
**Project Type:** Comparative Machine Learning Study  
**Dataset:** Wisconsin Breast Cancer Dataset

---

## 1. Problem Statement

Breast cancer is one of the most common cancers affecting women worldwide, with early detection being critical for successful treatment and survival. Medical professionals rely on diagnostic imaging techniques such as Fine Needle Aspiration (FNA) to extract cells from suspicious breast masses for analysis. These cells are then evaluated based on various morphological characteristics to determine whether the tumor is **benign (non-cancerous)** or **malignant (cancerous)**.

### Core Challenges:

1. **High Dimensionality**: Medical diagnostic data contains numerous features (30 features in this dataset), many of which may be redundant, correlated, or irrelevant
2. **Model Performance**: Different machine learning algorithms have varying capabilities in handling medical classification tasks
3. **Feature Redundancy**: Using all available features can lead to:
   - Increased computational cost
   - Overfitting on training data
   - Reduced model interpretability
   - Longer training and prediction times
4. **Accuracy vs Efficiency Trade-off**: Finding the optimal balance between model accuracy and computational efficiency

### Research Question:

> **"Can Genetic Algorithm-based feature selection improve the performance, efficiency, and interpretability of machine learning models for breast cancer classification while maintaining or enhancing diagnostic accuracy?"**

This project addresses the need for **automated, accurate, and efficient** breast cancer classification systems that can assist medical professionals in making faster and more reliable diagnostic decisions.

---

## 2. Project Objectives

### Primary Objectives:

1. **Binary Classification Task**

   - Develop machine learning models to classify breast cancer tumors as:
     - **Benign (B)**: Non-cancerous, not life-threatening
     - **Malignant (M)**: Cancerous, requires immediate medical intervention

2. **Comparative Algorithm Analysis**

   - Implement and evaluate four different machine learning algorithms:
     - **Artificial Neural Network (ANN/MLP)**: Deep learning approach
     - **K-Nearest Neighbors (KNN)**: Instance-based learning
     - **Naive Bayes**: Probabilistic classifier
     - **Support Vector Machine (SVM)**: Margin-based classifier

3. **Feature Selection Using Genetic Algorithm**

   - Apply evolutionary optimization to identify the most relevant feature subset
   - Reduce dimensionality while preserving or improving classification accuracy
   - Demonstrate the effectiveness of bio-inspired optimization in feature engineering

4. **Performance Comparison**
   - Compare each algorithm's performance WITH and WITHOUT GA-selected features
   - Analyze the impact of dimensionality reduction on:
     - Accuracy
     - Precision
     - Recall (sensitivity)
     - F1 Score
     - Training time
     - Model complexity

### Secondary Objectives:

5. **Demonstrate Best Practices in ML Pipeline**

   - Data preprocessing and normalization
   - Proper train-test splitting
   - Reproducible experiments (random_state=42)
   - Comprehensive evaluation metrics

6. **Medical AI Considerations**

   - Emphasize **recall (sensitivity)** as a critical metric
   - Minimize false negatives (missing cancer cases)
   - Balance precision to avoid unnecessary patient anxiety

7. **Model Interpretability**
   - Identify which features are most important for diagnosis
   - Simplify models by reducing feature count
   - Provide insights into feature relevance

---

## 3. Dataset Description

### Dataset Source

**Wisconsin Diagnostic Breast Cancer (WDBC) Dataset**

- Original data contains 569 patient samples
- Features computed from digitized images of Fine Needle Aspirate (FNA) of breast mass
- Each sample represents measurements of cell nuclei characteristics

### Original Dataset: `data.csv`

#### Dataset Statistics:

| **Attribute**          | **Value**                               |
| ---------------------- | --------------------------------------- |
| **Total Samples**      | 569 patients                            |
| **Total Columns**      | 32 (ID + Diagnosis + 30 Features)       |
| **Features**           | 30 numerical features                   |
| **Target Variable**    | `diagnosis` (Categorical: M or B)       |
| **Class Distribution** | Malignant (M): 212 samples (37.3%)      |
|                        | Benign (B): 357 samples (62.7%)         |
| **Missing Values**     | None (handled by filling with 0 if any) |
| **Data Type**          | Continuous numerical values             |

#### Dataset Columns:

1. **`id`**: Patient identifier → **Dropped** during preprocessing
2. **`diagnosis`**: Target variable
   - `M` = Malignant (cancerous) → Encoded as **1**
   - `B` = Benign (non-cancerous) → Encoded as **0**
3. **`Unnamed: 32`**: Empty column → **Dropped**
4. **30 Feature Columns**: Organized in three groups

### Feature Categories

The 30 features describe characteristics of cell nuclei and are computed in three statistical aggregations:

#### 10 Base Measurements (for each, mean, SE, and worst are computed):

1. **Radius**: Mean distance from center to perimeter points
2. **Texture**: Standard deviation of gray-scale values
3. **Perimeter**: Circumference of the cell nucleus
4. **Area**: Surface area of the cell nucleus
5. **Smoothness**: Local variation in radius lengths
6. **Compactness**: (perimeter² / area) - 1.0
7. **Concavity**: Severity of concave portions of the contour
8. **Concave Points**: Number of concave portions of the contour
9. **Symmetry**: Symmetry of the cell nucleus
10. **Fractal Dimension**: "Coastline approximation" - 1

#### Feature Groups:

- **Mean values** (10 features): `radius_mean`, `texture_mean`, `perimeter_mean`, etc.
- **Standard Error** (10 features): `radius_se`, `texture_se`, `perimeter_se`, etc.
- **Worst/Largest** (10 features): `radius_worst`, `texture_worst`, `perimeter_worst`, etc.

**Total**: 30 numerical features

---

### Dataset After Genetic Algorithm Feature Selection

#### Reduced Dataset: `breast_cancer_selected_features.csv`

After applying the Genetic Algorithm for feature selection:

| **Attribute**                | **Value**                              |
| ---------------------------- | -------------------------------------- |
| **Total Samples**            | 569 (unchanged)                        |
| **Selected Features**        | **14 features** (+ 1 diagnosis column) |
| **Dimensionality Reduction** | **30 → 14 features (53.3% reduction)** |
| **Features Selected**        | (Determined by GA optimization)        |

#### Selected Features (by Genetic Algorithm):

The GA identified the following 14 most relevant features:

1. `texture_mean`
2. `area_mean`
3. `compactness_mean`
4. `concavity_mean`
5. `symmetry_mean`
6. `radius_se`
7. `texture_se`
8. `concavity_se`
9. `radius_worst`
10. `perimeter_worst`
11. `smoothness_worst`
12. `compactness_worst`
13. `concavity_worst`
14. `fractal_dimension_worst`

**Key Observations:**

- Mix of mean, standard error, and worst values
- Includes structural features (texture, area, compactness, concavity)
- Preserves shape characteristics (symmetry, fractal dimension)
- Includes boundary measurements (perimeter)

---

## 4. Dimension Reduction Impact

### Genetic Algorithm Feature Selection Process

#### GA Configuration:

```python
Population Size: 20 individuals
Generations: 30
Crossover Probability: 0.8 (80%)
Mutation Probability: 0.1 (10%)
Fitness Function: KNN Classification Accuracy
Selection Method: Roulette Wheel Selection
```

#### How GA Works:

1. **Initialization**: Creates 20 random binary vectors (length 30)

   - Each bit represents whether to include (1) or exclude (0) a feature

2. **Fitness Evaluation**:

   - For each individual, train a KNN classifier using selected features
   - Calculate accuracy on test set
   - Higher accuracy = better fitness

3. **Selection**:

   - Roulette wheel selection favors better individuals
   - Probability of selection proportional to fitness

4. **Crossover** (80% probability):

   - Two parent individuals exchange feature selections
   - Creates offspring with mixed feature combinations

5. **Mutation** (10% per bit):

   - Randomly flip feature inclusion/exclusion
   - Introduces diversity to prevent premature convergence

6. **Evolution**:
   - Repeats for 30 generations
   - Population improves over time
   - Best individual selected at the end

### Dimensionality Reduction Impact Analysis

#### Quantitative Impact:

| **Metric**                 | **Before GA** | **After GA** | **Change**    |
| -------------------------- | ------------- | ------------ | ------------- |
| **Number of Features**     | 30            | 14           | -16 (-53.3%)  |
| **Feature Space Size**     | 30D           | 14D          | 53% reduction |
| **Input Dimensionality**   | High          | Medium       | Reduced       |
| **Training Data Required** | More          | Less         | Reduced       |
| **Model Complexity**       | Higher        | Lower        | Simplified    |

#### Benefits of Dimensionality Reduction:

##### 1. **Computational Efficiency**

- **Faster Training**: Fewer features mean faster model convergence
- **Reduced Memory Usage**: Smaller feature matrices
- **Quicker Predictions**: Less computation during inference
- **Lower Storage**: Smaller model size

##### 2. **Reduced Overfitting Risk**

- Fewer features reduce the chance of memorizing noise
- Simpler models generalize better to unseen data
- Less prone to capturing spurious correlations

##### 3. **Improved Model Interpretability**

- 14 features are easier to understand than 30
- Medical professionals can focus on key diagnostic indicators
- Better explainability for clinical decision-making

##### 4. **Feature Relevance**

- Eliminates redundant features (e.g., highly correlated measurements)
- Removes noisy or irrelevant features
- Retains most informative features for classification

##### 5. **Curse of Dimensionality Mitigation**

- In high-dimensional spaces, data becomes sparse
- Distance-based algorithms (KNN, SVM) suffer in high dimensions
- Reduction to 14D improves algorithm effectiveness

### Feature Selection Quality

The GA-selected features represent a **balanced mix**:

- **Structural features**: area, compactness, concavity
- **Texture information**: texture_mean, texture_se
- **Shape characteristics**: symmetry, fractal_dimension
- **Size indicators**: radius, perimeter
- **Statistical variations**: standard error (se) and worst values

This demonstrates that the GA successfully identified features across different measurement categories, suggesting a comprehensive diagnostic profile.

### Comparison: Manual vs Automated Selection

| **Approach**           | **Manual Feature Selection** | **GA Feature Selection**        |
| ---------------------- | ---------------------------- | ------------------------------- |
| **Expertise Required** | Domain knowledge needed      | Automated, data-driven          |
| **Time Investment**    | Days/weeks                   | Hours (30 generations)          |
| **Optimization**       | May miss non-obvious combos  | Explores 20×30 = 600 solutions  |
| **Bias**               | Human bias present           | Objective, fitness-based        |
| **Reproducibility**    | Difficult                    | Reproducible (with random seed) |

---

## 5. Performance Evaluation

### Evaluation Metrics

All models are evaluated using standard classification metrics:

#### 1. **Accuracy**

- Percentage of correct predictions overall
- Formula: `(TP + TN) / (TP + TN + FP + FN)`
- Useful for balanced datasets

#### 2. **Precision**

- Of all predicted malignant cases, how many are truly malignant
- Formula: `TP / (TP + FP)`
- Important for minimizing false alarms

#### 3. **Recall (Sensitivity)**

- Of all actual malignant cases, how many were correctly detected
- Formula: `TP / (TP + FN)`
- **CRITICAL in medical diagnosis** - missing cancer is dangerous!

#### 4. **F1 Score**

- Harmonic mean of precision and recall
- Formula: `2 × (Precision × Recall) / (Precision + Recall)`
- Balances both precision and recall

#### 5. **Confusion Matrix**

```
                Predicted
                B       M
Actual    B   [TN]    [FP]
          M   [FN]    [TP]
```

- **TN** (True Negative): Correctly predicted benign
- **FP** (False Positive): Predicted malignant, actually benign (false alarm)
- **FN** (False Negative): Predicted benign, actually malignant ⚠️ DANGEROUS!
- **TP** (True Positive): Correctly predicted malignant

### Model Implementation Details

#### 1. **Artificial Neural Network (MLP)**

**Architecture:**

- Input Layer: 30 neurons (no GA) or 14 neurons (with GA)
- Hidden Layer 1: 16 neurons, ReLU activation
- Hidden Layer 2: 8 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation

**Training Configuration:**

- Optimizer: Adam (learning rate = 0.001)
- Loss Function: Binary Cross-Entropy
- Epochs: 100
- Batch Size: 16
- Validation Split: 10%

#### 2. **K-Nearest Neighbors (KNN)**

**Configuration:**

- `n_neighbors = 5`
- Distance Metric: Euclidean
- Weights: Uniform

**Characteristics:**

- Instance-based learning
- No explicit training phase
- Simple but effective

#### 3. **Naive Bayes**

**Configuration:**

- Type: Gaussian Naive Bayes
- Assumes features follow Gaussian distribution
- No hyperparameters to tune

**Characteristics:**

- Probabilistic classifier
- Assumes feature independence
- Fast training and prediction

#### 4. **Support Vector Machine (SVM)**

**Configuration:**

- Kernel: Radial Basis Function (RBF)
- Regularization Parameter (C): 1.0
- Gamma: 'scale'

**Characteristics:**

- Finds optimal separating hyperplane
- Effective in high-dimensional spaces
- Robust to overfitting

### Data Preprocessing

All models use consistent preprocessing:

1. **Label Encoding**:

   - Malignant (M) → 1
   - Benign (B) → 0

2. **Train-Test Split**:

   - Training Set: 80% (455 samples)
   - Test Set: 20% (114 samples)
   - Random State: 42 (for reproducibility)

3. **Feature Scaling**:
   - StandardScaler applied (except for baseline Naive Bayes)
   - Transforms features to mean=0, std=1
   - Critical for KNN, SVM, and ANN

### Expected Performance Comparison Framework

To evaluate the models, the following comparison should be performed:

#### Performance Comparison Table (Template):

| **Model**             | **Features** | **Accuracy** | **Precision** | **Recall** | **F1 Score** | **Training Time** |
| --------------------- | ------------ | ------------ | ------------- | ---------- | ------------ | ----------------- |
| ANN (No GA)           | 30           | ?            | ?             | ?          | ?            | ?                 |
| ANN (With GA)         | 14           | ?            | ?             | ?          | ?            | ?                 |
| KNN (No GA)           | 30           | ?            | ?             | ?          | ?            | ?                 |
| KNN (With GA)         | 14           | ?            | ?             | ?          | ?            | ?                 |
| Naive Bayes (No GA)   | 30           | ?            | ?             | ?          | ?            | ?                 |
| Naive Bayes (With GA) | 14           | ?            | ?             | ?          | ?            | ?                 |
| SVM (No GA)           | 30           | ?            | ?             | ?          | ?            | ?                 |
| SVM (With GA)         | 14           | ?            | ?             | ?          | ?            | ?                 |

**Note**: To populate this table, run each model script and record the output metrics.

### Performance Analysis Guidelines

When analyzing results, consider:

#### Key Questions:

1. **Does GA improve accuracy?**

   - Compare accuracy scores for each algorithm (GA vs No GA)
   - Is the feature reduction worth any accuracy trade-off?

2. **Which algorithm performs best?**

   - Look at overall accuracy and F1 scores
   - Consider which algorithm is most suitable for production

3. **Is recall maintained or improved?**

   - **Critical**: In cancer diagnosis, recall is more important than precision
   - Missing a malignant tumor (False Negative) is far worse than a false alarm

4. **Training time improvements?**

   - GA models should train faster due to fewer features
   - Quantify the speedup

5. **Model simplicity vs performance?**
   - Did simpler models (14 features) achieve comparable results?
   - Is interpretability improved?

### Medical Performance Considerations

In a medical diagnostic context:

- **Recall (Sensitivity) > Precision**

  - Missing cancer is life-threatening
  - False alarms can be verified with additional tests

- **Target Metrics**:
  - Recall: Ideally > 95% (catch most malignant cases)
  - Precision: > 85% (minimize unnecessary worry)
  - F1 Score: > 90% (balanced performance)

---

## 6. Experimental Setup and Reproducibility

### Environment Requirements

```bash
# Python packages
pandas          # Data manipulation
numpy           # Numerical operations
scikit-learn    # ML algorithms and utilities
tensorflow      # Deep learning (ANN)
```

### Installation

```bash
pip install pandas numpy scikit-learn tensorflow
```

### Execution Workflow

#### Step 1: Feature Selection (One-time)

```bash
python src/genetic_algorithm.py
```

**Output**: `breast_cancer_selected_features.csv`

#### Step 2: Train and Evaluate Models

**Models WITHOUT GA (Baseline):**

```bash
python src/ann_no_GA.py
python src/knn_no_GA.py
python src/naive_bayes_no_GA.py
python src/support_vector_machine_no_GA.py
```

**Models WITH GA (Optimized):**

```bash
python src/ann_GA.py
python src/knn_GA.py
python src/naive_bayes_GA.py
python src/support_vector_machine_GA.py
```

#### Step 3: Analyze and Compare Results

- Document all metrics in a comparison table
- Create visualizations (bar charts, confusion matrices)
- Draw conclusions about GA effectiveness

---

## 7. Project Structure

```
Advanced_AI_Proj/
│
├── src/
│   ├── data_set/
│   │   ├── data.csv                              # Original dataset (569×32)
│   │   ├── breast_cancer_selected_features.csv   # GA-selected (569×15)
│   │   └── train.csv                             # Additional training data
│   │
│   ├── genetic_algorithm.py                      # GA feature selection
│   ├── processing.py                             # Data exploration (original)
│   ├── processing_new_data_set.py                # Data exploration (GA dataset)
│   │
│   ├── ann_GA.py                                 # ANN with GA features
│   ├── ann_no_GA.py                              # ANN with all features
│   │
│   ├── knn_GA.py                                 # KNN with GA features
│   ├── knn_no_GA.py                              # KNN with all features
│   │
│   ├── naive_bayes_GA.py                         # Naive Bayes with GA
│   ├── naive_bayes_no_GA.py                      # Naive Bayes without GA
│   │
│   ├── support_vector_machine_GA.py              # SVM with GA
│   └── support_vector_machine_no_GA.py           # SVM without GA
│
├── PROJECT_DOCUMENTATION.md                      # Detailed technical docs
├── PROJECT_REPORT.md                             # This report
└── README.md                                     # Quick start guide
```

---

## 8. Key Findings and Expected Outcomes

### Hypotheses:

1. **Hypothesis 1**: GA will select 10-15 most relevant features

   - ✅ **Confirmed**: 14 features selected (53% reduction)

2. **Hypothesis 2**: GA models will train faster

   - ⏳ **To be verified**: Compare training times

3. **Hypothesis 3**: GA models will maintain or improve accuracy

   - ⏳ **To be verified**: Run all experiments

4. **Hypothesis 4**: Different algorithms will have different optimal feature sets
   - Note: Current implementation uses one GA run for all models
   - Future work: Algorithm-specific GA optimization

### Expected Results:

Based on machine learning theory and previous studies:

- **KNN**: Expected to benefit most from dimensionality reduction

  - Curse of dimensionality affects distance-based algorithms
  - Faster computation with fewer features

- **SVM**: Should perform well with or without GA

  - Already handles high dimensions effectively
  - Feature reduction may improve speed

- **ANN**: May have slight accuracy trade-off

  - Neural networks can learn feature importance
  - But faster training expected with 14 features

- **Naive Bayes**: Variable performance
  - Assumes feature independence (violated with correlated features)
  - GA may help by removing redundant features

---

## 9. Limitations and Future Work

### Current Limitations:

1. **Single Feature Set**:

   - GA optimization performed once using KNN as fitness function
   - Same 14 features used for all algorithms
   - Each algorithm may benefit from different feature subsets

2. **Fixed GA Parameters**:

   - Population size, generations, crossover/mutation rates are constants
   - No hyperparameter tuning performed

3. **Single Train-Test Split**:

   - Results based on one 80-20 split
   - More robust evaluation would use cross-validation

4. **Limited Fitness Functions**:

   - GA uses only accuracy
   - Could optimize for F1 score, recall, or multi-objective fitness

5. **No Ensemble Methods**:
   - Individual classifiers only
   - No voting, bagging, or boosting explored

### Future Enhancements:

1. **Algorithm-Specific Feature Selection**

   - Run separate GA for each algorithm
   - Use respective algorithm as fitness evaluator
   - Expected result: Different optimal feature subsets

2. **Multi-Objective Optimization**

   - Optimize for both accuracy AND feature count
   - Pareto front of solutions
   - Trade-off curve: accuracy vs complexity

3. **Cross-Validation**

   - 5-fold or 10-fold cross-validation
   - More reliable performance estimates
   - Reduce variance in results

4. **Hyperparameter Tuning**

   - GridSearchCV or RandomizedSearchCV
   - Optimize algorithm parameters
   - Could further improve performance

5. **Advanced Feature Selection Methods**

   - Compare GA with other methods:
     - Recursive Feature Elimination (RFE)
     - LASSO regularization
     - Tree-based feature importance
     - Principal Component Analysis (PCA)

6. **Ensemble Methods**

   - Voting classifier combining all 4 algorithms
   - Bagging and boosting techniques
   - Expected improvement in robustness

7. **Explainable AI (XAI)**

   - SHAP values for feature importance
   - LIME for local explanations
   - Better clinical interpretability

8. **Deployment Considerations**
   - Model serialization (save/load)
   - REST API for predictions
   - Web interface for clinical use
   - Real-time prediction system

---

## 10. Conclusions

### Project Accomplishments:

This project successfully demonstrates:

✅ **Complete ML Pipeline**: From raw data to trained models  
✅ **Evolutionary Optimization**: GA for intelligent feature selection  
✅ **Comparative Analysis**: 8 different model configurations (4 algorithms × 2 versions)  
✅ **Dimensionality Reduction**: 53% feature reduction (30 → 14)  
✅ **Reproducible Science**: Fixed random seeds, documented methodology  
✅ **Medical AI Context**: Appropriate metrics and considerations

### Impact of Genetic Algorithm:

The GA-based feature selection provides:

1. **Efficiency**: Reduced feature space by 53%
2. **Automation**: No manual feature engineering required
3. **Objectivity**: Data-driven, not based on human bias
4. **Optimization**: Explores hundreds of feature combinations

### Practical Applications:

This research has implications for:

- **Clinical Decision Support**: Faster, automated cancer screening
- **Medical Imaging**: Identifying key diagnostic features
- **Healthcare AI**: Template for other diagnostic tasks
- **Feature Engineering**: Demonstrating GA effectiveness

### Educational Value:

Perfect for learning:

- Machine learning fundamentals
- Feature selection techniques
- Genetic algorithms
- Model comparison methodology
- Medical AI considerations

---

## 11. References and Resources

### Dataset:

- **Wisconsin Diagnostic Breast Cancer Dataset**
  - UCI Machine Learning Repository
  - Creators: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

### Algorithms:

- **K-Nearest Neighbors**: Cover & Hart (1967)
- **Naive Bayes**: Based on Bayes' Theorem
- **Support Vector Machines**: Cortes & Vapnik (1995)
- **Artificial Neural Networks**: Backpropagation (Rumelhart et al., 1986)
- **Genetic Algorithms**: Holland (1975)

### Libraries:

- **scikit-learn**: Pedregosa et al. (2011)
- **TensorFlow/Keras**: Abadi et al. (2015)
- **pandas**: McKinney (2010)
- **NumPy**: Harris et al. (2020)

---

## 12. Appendix

### A. Running Individual Models

Each model script can be run independently:

```bash
# Example: Run KNN with GA features
cd src
python knn_GA.py
```

**Expected Output:**

```
KNN Results on GA-Selected Breast Cancer Dataset:
Accuracy : 0.XXXX
Precision: 0.XXXX
Recall   : 0.XXXX
F1 Score : 0.XXXX

Confusion Matrix:
[[TN FP]
 [FN TP]]
```

### B. Modifying GA Parameters

Edit `genetic_algorithm.py`:

```python
# Line 24-27
population_size = 20      # Try: 30, 50, 100
num_generations = 30      # Try: 50, 100, 200
crossover_prob = 0.8      # Try: 0.6, 0.7, 0.9
mutation_prob = 0.1       # Try: 0.05, 0.15, 0.2
```

### C. Medical AI Ethics

⚠️ **Important Disclaimer**:

This project is for **research and educational purposes only**.

- **NOT approved for clinical use**
- **NOT a substitute for professional medical diagnosis**
- Real-world medical AI requires:
  - Regulatory approval (FDA, EMA, etc.)
  - Clinical validation studies
  - Much larger, diverse datasets
  - Expert medical oversight
  - Rigorous testing and monitoring

---

**Report Prepared By:** Advanced AI Project Team  
**Last Updated:** December 5, 2025  
**Project Status:** Completed - Ready for Experimentation  
**License:** Educational Use Only

---

### Contact and Contribution

For questions, improvements, or contributions:

- Follow the coding conventions in existing files
- Maintain reproducibility (use `random_state=42`)
- Document all changes
- Report comprehensive metrics

---

**End of Report**
