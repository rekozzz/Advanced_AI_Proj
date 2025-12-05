# Breast Cancer Classification with Genetic Algorithm Feature Selection

## 📋 Project Overview

This project is a **comprehensive machine learning study** for **breast cancer classification** (Benign vs Malignant). The main goal is to compare the performance of four different machine learning algorithms **with and without Genetic Algorithm (GA) feature selection** to determine if GA-based feature optimization improves model accuracy and efficiency.

### 🎯 Primary Objectives

1. **Classify breast cancer tumors** as Benign (B) or Malignant (M) using medical diagnostic features
2. **Use Genetic Algorithm** to select the most important features from the dataset
3. **Compare performance** of machine learning models with all features vs GA-selected features
4. **Evaluate multiple algorithms**: Artificial Neural Network (ANN/MLP), K-Nearest Neighbors (KNN), Naive Bayes, and Support Vector Machine (SVM)

---

## 📂 Project Structure

```
Advanced_AI_Proj/
├── data.csv                              # Original breast cancer dataset (569 samples, 30 features)
├── train.csv                             # Additional training dataset
├── breast_cancer_selected_features.csv   # Dataset with GA-selected features only
│
├── genetic_algorithm.py                  # GA implementation for feature selection
├── processing.py                         # Data exploration for original dataset
├── processing_new_data_set.py            # Data exploration for GA-selected dataset
│
├── ann_GA.py                             # Neural Network WITH GA-selected features
├── ann_no_GA.py                          # Neural Network WITHOUT GA (all features)
│
├── knn_GA.py                             # K-Nearest Neighbors WITH GA-selected features
├── knn_no_GA.py                          # K-Nearest Neighbors WITHOUT GA (all features)
│
├── naive_bayes_GA.py                     # Naive Bayes WITH GA-selected features
├── naive_bayes_no_GA.py                  # Naive Bayes WITHOUT GA (all features)
│
├── support_vector_machine_GA.py          # SVM WITH GA-selected features
└── support_vector_machine_no_GA.py       # SVM WITHOUT GA (all features)
```

---

## 📊 Dataset Description

### Source Data: `data.csv`

The dataset contains **569 patient samples** with **32 columns**:

- **`id`**: Patient identifier (dropped during preprocessing)
- **`diagnosis`**: Target variable
  - `M` = Malignant (cancerous) → encoded as `1`
  - `B` = Benign (non-cancerous) → encoded as `0`
- **30 numerical features** computed from digitized images of fine needle aspirate (FNA) of breast mass

### Feature Categories

Features are computed for three aspects of cell nuclei:

1. **Mean values** (10 features): radius_mean, texture_mean, perimeter_mean, etc.
2. **Standard error** (10 features): radius_se, texture_se, perimeter_se, etc.
3. **Worst/largest values** (10 features): radius_worst, texture_worst, perimeter_worst, etc.

**Total**: 30 numerical features describing characteristics like:

- Radius, Texture, Perimeter, Area
- Smoothness, Compactness, Concavity
- Concave points, Symmetry, Fractal dimension

---

## 🧬 How the Project Works

### Step 1: Genetic Algorithm Feature Selection

**File**: `genetic_algorithm.py`

This is the **core innovation** of the project. Instead of using all 30 features, the GA intelligently selects the most relevant subset.

#### GA Process:

1. **Initialization**: Creates 20 random individuals (binary vectors of length 30)

   - Each bit represents whether to include a feature (1) or exclude it (0)
   - Example: `[1, 0, 1, 1, 0, ...]` means use features 1, 3, 4, but not 2, 5, etc.

2. **Fitness Evaluation**: For each individual:

   - Select only the features where bit = 1
   - Train a KNN classifier on those features
   - Calculate accuracy on test set
   - Higher accuracy = better fitness

3. **Selection**: Uses **Roulette Wheel Selection**

   - Individuals with higher fitness have higher probability of being selected
   - Better solutions are more likely to pass their "genes" to next generation

4. **Crossover** (80% probability):

   - Two parent individuals exchange parts of their binary vectors
   - Creates two offspring with mixed feature selections
   - Example:
     - Parent1: `[1,1,0,0,1]` + Parent2: `[0,0,1,1,0]`
     - Offspring: `[1,1,1,1,0]` and `[0,0,0,0,1]`

5. **Mutation** (10% probability per bit):

   - Randomly flip bits (0→1 or 1→0)
   - Introduces diversity and prevents premature convergence

6. **Evolution**: Repeats for **30 generations**

7. **Output**:
   - Best feature subset selected
   - Creates `breast_cancer_selected_features.csv` with only selected features

#### GA Parameters:

```python
population_size = 20        # Number of solutions in each generation
num_generations = 30        # Number of evolutionary cycles
crossover_prob = 0.8        # 80% chance of crossover
mutation_prob = 0.1         # 10% chance of bit flip per feature
```

---

### Step 2: Data Processing & Exploration

#### For Original Dataset: `processing.py`

- Loads `data.csv`
- Drops `id` and `Unnamed: 32` columns
- Encodes diagnosis: M→1, B→0
- Fills missing values with 0
- Splits into 80% train / 20% test
- Displays data info and shapes

#### For GA-Selected Dataset: `processing_new_data_set.py`

- Same process but for `breast_cancer_selected_features.csv`
- Also applies StandardScaler for feature normalization
- Shows reduced number of features after GA selection

---

### Step 3: Machine Learning Model Training & Evaluation

The project implements **4 different ML algorithms**, each with **2 versions**:

#### 1️⃣ Artificial Neural Network (MLP - Multi-Layer Perceptron)

**Files**: `ann_GA.py` and `ann_no_GA.py`

**Architecture**:

```
Input Layer → 16 neurons (ReLU) → 8 neurons (ReLU) → 1 neuron (Sigmoid) → Output
```

**Details**:

- Uses TensorFlow/Keras
- Binary cross-entropy loss
- Adam optimizer (learning rate = 0.001)
- Trains for 100 epochs with batch size 16
- 10% validation split during training
- StandardScaler for feature normalization

**Difference**:

- `ann_GA.py`: Uses GA-selected features → fewer input neurons
- `ann_no_GA.py`: Uses all 30 features → more input neurons

---

#### 2️⃣ K-Nearest Neighbors (KNN)

**Files**: `knn_GA.py` and `knn_no_GA.py`

**Configuration**:

- `n_neighbors = 5` (considers 5 closest training samples)
- Uses Euclidean distance
- StandardScaler for feature normalization

**How it works**:

- For each test sample, finds 5 nearest training samples
- Majority vote determines classification
- Simple but effective for this type of data

---

#### 3️⃣ Naive Bayes

**Files**: `naive_bayes_GA.py` and `naive_bayes_no_GA.py`

**Configuration**:

- Uses Gaussian Naive Bayes
- Assumes features follow Gaussian (normal) distribution
- No hyperparameters to tune

**How it works**:

- Calculates probability of each class given feature values
- Uses Bayes' theorem
- Assumes feature independence (naive assumption)
- Fast training and prediction

---

#### 4️⃣ Support Vector Machine (SVM)

**Files**: `support_vector_machine_GA.py` and `support_vector_machine_no_GA.py`

**Configuration**:

- Kernel: Radial Basis Function (RBF)
- `C = 1.0` (regularization parameter)
- `gamma = 'scale'` (kernel coefficient)
- StandardScaler for feature normalization

**How it works**:

- Finds optimal hyperplane to separate classes
- Uses RBF kernel for non-linear decision boundaries
- Effective in high-dimensional spaces

---

## 📈 Performance Metrics

All models are evaluated using:

1. **Accuracy**: Overall correctness

   - Formula: `(TP + TN) / (TP + TN + FP + FN)`

2. **Precision**: Of predicted malignant cases, how many are actually malignant

   - Formula: `TP / (TP + FP)`
   - Important: Minimizes false alarms

3. **Recall (Sensitivity)**: Of actual malignant cases, how many we detected

   - Formula: `TP / (TP + FN)`
   - Critical: Missing cancer is dangerous!

4. **F1 Score**: Harmonic mean of precision and recall

   - Formula: `2 * (Precision * Recall) / (Precision + Recall)`
   - Balances both metrics

5. **Confusion Matrix**: Detailed breakdown
   ```
   [[TN  FP]
    [FN  TP]]
   ```
   - TN = True Negative (correctly predicted benign)
   - FP = False Positive (predicted malignant, actually benign)
   - FN = False Negative (predicted benign, actually malignant) ⚠️ DANGEROUS!
   - TP = True Positive (correctly predicted malignant)

---

## 🚀 How to Run the Project

### Prerequisites

Install required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow
```

### Execution Order

#### Step 1: Run Genetic Algorithm (One-time)

```bash
python genetic_algorithm.py
```

**Output**: Creates `breast_cancer_selected_features.csv` with optimized feature subset

#### Step 2: Data Exploration (Optional)

```bash
python processing.py                    # Explore original dataset
python processing_new_data_set.py       # Explore GA-selected dataset
```

#### Step 3: Train & Evaluate Models

**Without GA (baseline)**:

```bash
python ann_no_GA.py
python knn_no_GA.py
python naive_bayes_no_GA.py
python support_vector_machine_no_GA.py
```

**With GA (optimized)**:

```bash
python ann_GA.py
python knn_GA.py
python naive_bayes_GA.py
python support_vector_machine_GA.py
```

#### Step 4: Compare Results

- Review accuracy, precision, recall, and F1 scores
- Determine which algorithm performs best
- Assess impact of GA feature selection

---

## 🔬 Expected Workflow for New Developers

### Understanding the Comparison

This project answers the question: **"Does reducing features with GA improve model performance?"**

| Aspect                  | WITHOUT GA             | WITH GA                              |
| ----------------------- | ---------------------- | ------------------------------------ |
| **Features**            | All 30 features        | GA-selected subset (~10-15 features) |
| **Training Time**       | Slower (more features) | Faster (fewer features)              |
| **Model Complexity**    | Higher                 | Lower                                |
| **Risk of Overfitting** | Higher                 | Lower                                |
| **Accuracy**            | ? (to be measured)     | ? (to be measured)                   |

### Making Modifications

#### To change GA parameters:

Edit `genetic_algorithm.py`:

```python
population_size = 20      # Try: 30, 50
num_generations = 30      # Try: 50, 100
crossover_prob = 0.8      # Try: 0.7, 0.9
mutation_prob = 0.1       # Try: 0.05, 0.15
```

#### To change model hyperparameters:

**KNN**: Change `n_neighbors` in KNN files

```python
model = KNeighborsClassifier(n_neighbors=5)  # Try: 3, 7, 10
```

**SVM**: Change kernel or C parameter

```python
model = SVC(kernel='rbf', C=1.0)  # Try: kernel='linear', C=0.1 or C=10
```

**ANN**: Modify network architecture

```python
model.add(Dense(16, activation='relu'))  # Try: 32, 64 neurons
model.add(Dense(8, activation='relu'))   # Try: 16, 32 neurons
```

#### To use different dataset split:

Change `test_size` in all files:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # Try: 0.3, 0.25
)
```

---

## 🔍 Key Concepts Explained

### What is Feature Selection?

**Problem**: Not all 30 features may be equally important. Some might be:

- Redundant (highly correlated with others)
- Noisy (add randomness without information)
- Irrelevant (don't help distinguish benign from malignant)

**Solution**: Use GA to automatically find the best subset that maximizes accuracy while minimizing feature count.

**Benefits**:

- ✅ Faster training and prediction
- ✅ Reduced overfitting
- ✅ Better interpretability
- ✅ Lower computational cost
- ✅ Potentially higher accuracy

### Why Genetic Algorithm?

GA is inspired by biological evolution:

- **Individuals** = Possible feature subsets
- **Genes** = Individual features (included or excluded)
- **Fitness** = Model accuracy on that feature subset
- **Natural selection** = Better subsets survive and reproduce
- **Evolution** = Population improves over generations

**Advantages over other methods**:

- Explores many combinations simultaneously
- Avoids getting stuck in local optima
- Doesn't require assumptions about feature relationships
- Can find non-obvious feature combinations

---

## 📝 Common Data Science Notes

### Train-Test Split

- **Training set (80%)**: Used to learn patterns
- **Test set (20%)**: Used to evaluate performance on unseen data
- **random_state=42**: Ensures reproducible results across runs

### Feature Scaling (StandardScaler)

Transforms features to have:

- Mean = 0
- Standard deviation = 1

**Why needed**:

- KNN, SVM, ANN are sensitive to feature scales
- Prevents features with large ranges from dominating
- Example: "area" (range: 100-2000) vs "smoothness" (range: 0.05-0.15)

**Not needed for**:

- Naive Bayes (works with probabilities)

### Label Encoding

Converts categorical target to numerical:

- `M` (Malignant) → `1`
- `B` (Benign) → `0`

Required because ML algorithms work with numbers, not strings.

---

## 🎓 For New Developers: Next Steps

### 1. **Run Experiments**

- Execute all scripts and document results
- Create a comparison table of all 8 models

### 2. **Analyze Results**

- Which algorithm performs best?
- Does GA improve performance?
- Which features were selected by GA?

### 3. **Potential Improvements**

- Try different GA fitness functions (e.g., F1 score instead of accuracy)
- Implement cross-validation for more robust evaluation
- Try ensemble methods (combine multiple models)
- Add hyperparameter tuning (GridSearchCV)
- Visualize decision boundaries
- Create ROC curves
- Implement SHAP or LIME for feature importance explanation

### 4. **Create Reports**

- Document all results in a spreadsheet or markdown file
- Create visualizations (confusion matrices, accuracy comparisons)
- Write conclusions about effectiveness of GA

### 5. **Code Improvements**

- Add command-line arguments for easy parameter changes
- Create a main script that runs all experiments automatically
- Add logging instead of print statements
- Implement model saving/loading
- Add data validation and error handling

---

## ⚠️ Important Notes

### Medical Context

- This is a **research/educational project**
- **NOT intended for actual medical diagnosis**
- Real medical AI systems require:
  - Regulatory approval (FDA, etc.)
  - Much larger datasets
  - Clinical validation
  - Expert oversight

### Data Consistency

- All scripts use `random_state=42` for reproducibility
- Same train-test split ensures fair comparison
- Always run `genetic_algorithm.py` first before GA-based models

### File Dependencies

```
genetic_algorithm.py → breast_cancer_selected_features.csv → *_GA.py files
data.csv → *_no_GA.py files
```

---

## 🤝 Contributing

When adding new features or models:

1. Follow the naming convention: `algorithm_name_GA.py` and `algorithm_name_no_GA.py`
2. Use the same train-test split (`test_size=0.2, random_state=42`)
3. Report all 5 metrics (accuracy, precision, recall, F1, confusion matrix)
4. Document parameter choices
5. Add comments explaining the model

---

## 📚 References

### Libraries Used

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms (KNN, SVM, Naive Bayes) and utilities
- **tensorflow/keras**: Deep learning (ANN)

### Algorithms

- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier
- **Support Vector Machine**: Margin-based classifier
- **Artificial Neural Network**: Deep learning approach
- **Genetic Algorithm**: Evolutionary optimization

---

## 📞 Summary

This project provides a **complete comparative study** of machine learning approaches for breast cancer classification, with a focus on the impact of Genetic Algorithm feature selection. It demonstrates:

✅ Full ML pipeline (data loading → preprocessing → training → evaluation)  
✅ Multiple algorithm implementations  
✅ Feature engineering with evolutionary algorithms  
✅ Proper evaluation methodology  
✅ Clean, modular code structure

**Perfect for**: Learning ML, understanding GA, comparing algorithms, or as a foundation for more advanced research.

---

**Last Updated**: December 2025  
**Project Type**: Machine Learning Comparative Study  
**Domain**: Healthcare / Cancer Detection  
**Difficulty Level**: Intermediate to Advanced
