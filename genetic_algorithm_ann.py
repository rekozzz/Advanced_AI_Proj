import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# Fix random seeds for reproducibility
# ---------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

import tensorflow as tf
tf.random.set_seed(seed)

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("data.csv")
data = data.drop(columns=["id", "Unnamed: 32"], errors='ignore')

X = data.drop(columns=["diagnosis"]).values
y = LabelEncoder().fit_transform(data["diagnosis"])  # B->0, M->1

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

num_features = X.shape[1]

# ---------------------------
# GA Parameters
# ---------------------------
population_size = 30
num_generations = 30
crossover_prob = 0.8
mutation_prob = 0.1

# Initialize population: binary vectors (1=feature selected, 0=not selected)
population = np.random.randint(0, 2, size=(population_size, num_features))

# ---------------------------
# Fitness Function with Cross-Validation
# ---------------------------
def fitness(individual):
    if np.sum(individual) == 0:
        return 0  # avoid empty feature sets

    selected_indices = np.where(individual == 1)[0]
    X_sel = X[:, selected_indices]

    # Use Stratified K-Fold for stable evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in skf.split(X_sel, y):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = MLPClassifier(hidden_layer_sizes=(16, 8),
                              max_iter=400,
                              learning_rate_init=0.001,
                              random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    return np.mean(scores)  # average accuracy across folds

# ---------------------------
# Selection (Roulette Wheel)
# ---------------------------
def select(population, fitnesses):
    fitness_sum = np.sum(fitnesses)
    if fitness_sum == 0:
        return population.copy()  # if all fitnesses are 0
    probs = fitnesses / fitness_sum
    idx = np.random.choice(np.arange(population_size), size=population_size, p=probs)
    return population[idx]

# ---------------------------
# Crossover
# ---------------------------
def crossover(p1, p2):
    if random.random() < crossover_prob:
        point = random.randint(1, num_features-1)
        c1 = np.concatenate([p1[:point], p2[point:]])
        c2 = np.concatenate([p2[:point], p1[point:]])
        return c1, c2
    return p1.copy(), p2.copy()

# ---------------------------
# Mutation
# ---------------------------
def mutate(individual):
    for i in range(num_features):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]
    return individual

# ---------------------------
# GA Main Loop
# ---------------------------
for gen in range(num_generations):
    fitnesses = np.array([fitness(ind) for ind in population])
    print(f"Generation {gen+1}: Best Fitness = {fitnesses.max():.4f}")
    
    selected = select(population, fitnesses)
    
    next_pop = []
    for i in range(0, population_size, 2):
        p1, p2 = selected[i], selected[i+1]
        c1, c2 = crossover(p1, p2)
        next_pop.extend([mutate(c1), mutate(c2)])
    
    population = np.array(next_pop)

# ---------------------------
# Get best individual
# ---------------------------
fitnesses = np.array([fitness(ind) for ind in population])
best_idx = fitnesses.argmax()
best_individual = population[best_idx]

selected_feature_names = data.drop(columns=["diagnosis"]).columns[best_individual == 1].tolist()

print("\nSelected Features (for ANN):")
print(selected_feature_names)

# ---------------------------
# Save optimized dataset
# ---------------------------
new_data = data[selected_feature_names + ["diagnosis"]]
new_data.to_csv("new_breast_cancer_features_for_ann.csv", index=False)

