import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("data.csv")
data = data.drop(columns=["id", "Unnamed: 32"])

X = data.drop(columns=["diagnosis"]).values
y = data["diagnosis"].values

# Encode target: B=0, M=1
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]

# ---------------------------
# GA Parameters
# ---------------------------
population_size = 20
num_generations = 25
crossover_prob = 0.8
mutation_prob = 0.1

# Initial population (binary feature masks)
population = np.random.randint(0, 2, size=(population_size, num_features))

# ---------------------------
# Fitness Function (ANN-like)
# ---------------------------
def fitness(individual):
    # Avoid empty feature sets
    if np.sum(individual) == 0:
        return 0

    selected = np.where(individual == 1)[0]

    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[:, selected]

    # Fast ANN-like model (similar behavior to Keras MLP)
    model = MLPClassifier(hidden_layer_sizes=(16, 8),
                          max_iter=400,
                          learning_rate_init=0.001,
                          random_state=42)

    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    return accuracy_score(y_test, y_pred)

# ---------------------------
# Selection (Roulette Wheel)
# ---------------------------
def select(population, fitnesses):
    fitness_sum = np.sum(fitnesses)
    probs = fitnesses / fitness_sum
    idx = np.random.choice(population_size, size=population_size, p=probs)
    return population[idx]

# ---------------------------
# Crossover
# ---------------------------
def crossover(p1, p2):
    if random.random() < crossover_prob:
        point = random.randint(1, num_features - 1)
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
        next_pop.append(mutate(c1))
        next_pop.append(mutate(c2))

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
new_data.to_csv("breast_cancer_features_for_ann.csv", index=False)

print("\nNew optimized dataset saved as: breast_cancer_features_for_ann.csv")
