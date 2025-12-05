import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore') 

# -----------------------------
# Step 1: Load & Preprocess
# -----------------------------
# utilizing a dummy load for demonstration; replace with your actual csv load   
# data = pd.read_csv("breast_cancer.csv") 
from sklearn.datasets import load_breast_cancer
data_obj = load_breast_cancer()
X = pd.DataFrame(data_obj.data, columns=data_obj.feature_names)
y = data_obj.target

# -----------------------------
# Step 2: Split Data (Prevent Leakage)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CRITICAL: Scale data based ONLY on training set
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# -----------------------------
# Step 3: Genetic Algorithm
# -----------------------------
num_features = X_train.shape[1]
population_size = 10 # Reduced for speed in this demo
num_generations = 5
mutation_rate = 0.1

def initialize_population():
    return np.random.randint(2, size=(population_size, num_features))

def fitness(chromosome):
    # If no feature selected, return 0 fitness
    if chromosome.sum() == 0:
        return 0
    
    # Get column names based on the binary mask
    cols = X_train.columns
    selected_features = [cols[i] for i in range(len(chromosome)) if chromosome[i] == 1]
    
    # Define the model
    # Note: increased max_iter to ensure convergence
    model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    
    # --- THE FIX: USE CROSS-VALIDATION ---
    # We use 3-fold CV. This splits X_train into mini train/val sets 3 times.
    # The score is the average accuracy on unseen 'validation' slices.
    scores = cross_val_score(model, X_train_scaled[selected_features], y_train, cv=3)
    
    return scores.mean()

def select_parents(population, fitness_scores):
    # Added a check to prevent division by zero if all fitnesses are 0
    fitness_scores = np.array(fitness_scores)
    total_fitness = fitness_scores.sum()
    if total_fitness == 0:
        probs = None # Uniform probability
    else:
        probs = fitness_scores / total_fitness
        
    indices = np.random.choice(len(population), size=2, p=probs)
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2):
    point = np.random.randint(1, num_features-1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(chromosome):
    for i in range(num_features):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# --- Run the GA ---
population = initialize_population()

for generation in range(num_generations):
    fitness_scores = [fitness(chrom) for chrom in population]
    
    # Print progress
    best_gen_fitness = max(fitness_scores)
    print(f"Gen {generation+1}: Best CV Accuracy = {best_gen_fitness:.4f}")
    
    new_population = []
    
    # Elitism: Keep the absolute best parent automatically
    best_idx = np.argmax(fitness_scores)
    new_population.append(population[best_idx]) 

    while len(new_population) < population_size:
        p1, p2 = select_parents(population, fitness_scores)
        c1, c2 = crossover(p1, p2)
        new_population.append(mutate(c1))
        if len(new_population) < population_size:
            new_population.append(mutate(c2))

    population = np.array(new_population)

# -----------------------------
# Step 4: Evaluate Best Solution
# -----------------------------
final_fitness_scores = [fitness(chrom) for chrom in population]
best_index = np.argmax(final_fitness_scores)
best_chromosome = population[best_index]

# Convert binary mask to column names
cols = X_train.columns
final_features = [cols[i] for i in range(len(best_chromosome)) if best_chromosome[i] == 1]

print("\n--------------------------------")
print(f"Selected {len(final_features)} features: {final_features}")

# Train final model on ALL of X_train with selected features
final_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
final_model.fit(X_train_scaled[final_features], y_train)

# Test on X_test (The dataset the GA never saw)
y_pred_test = final_model.predict(X_test_scaled[final_features])
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Final Test Accuracy with Selected Features: {test_acc:.4f}")
# -----------------------------
# Step 5: Save GA-Selected Dataset
# -----------------------------

# Combine features + target for training + test sets
# Note: Using the original (unscaled) values for clarity
selected_data = pd.concat([
    X[final_features],
    pd.Series(y, name='diagnosis')  # original target
], axis=1)

# Save to CSV
selected_data.to_csv("breast_cancer_ga_selected_new_ann.csv", index=False)
print(f"✅ GA-selected dataset saved as 'breast_cancer_ga_selected.csv'")
