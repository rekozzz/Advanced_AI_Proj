import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

# Load your dataset
data = pd.read_csv("data.csv")

# Drop unnecessary columns: 'id' and empty 'Unnamed: 32'
data = data.drop(columns=['id', 'Unnamed: 32'])

# Separate features and target
X = data.drop(columns=['diagnosis']).values  # ONLY features
y = data['diagnosis'].values                # Target column

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# GA parameters
population_size = 20
num_generations = 30
crossover_prob = 0.8
mutation_prob = 0.1

num_features = X_train.shape[1]

# Initialize population: each individual is a binary vector representing feature selection
population = np.random.randint(0, 2, size=(population_size, num_features))

def fitness(individual):
    # If no feature is selected, return 0 fitness
    if np.sum(individual) == 0:
        return 0
    selected_indices = np.where(individual == 1)[0]
    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    
    model = KNeighborsClassifier()
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    return accuracy_score(y_test, y_pred)

def select(population, fitnesses):
    # Roulette wheel selection
    fitness_sum = np.sum(fitnesses)
    probs = fitnesses / fitness_sum
    selected_indices = np.random.choice(np.arange(population_size), size=population_size, p=probs)
    return population[selected_indices]

def crossover(parent1, parent2):
    if random.random() < crossover_prob:
        point = random.randint(1, num_features-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(individual):
    for i in range(num_features):
        if random.random() < mutation_prob:
            individual[i] = 1 - individual[i]
    return individual

# GA loop
for gen in range(num_generations):
    fitnesses = np.array([fitness(ind) for ind in population])
    print(f"Generation {gen+1}: Best Fitness = {fitnesses.max():.4f}")
    
    selected = select(population, fitnesses)
    next_population = []
    for i in range(0, population_size, 2):
        p1, p2 = selected[i], selected[i+1]
        c1, c2 = crossover(p1, p2)
        next_population.extend([mutate(c1), mutate(c2)])
    population = np.array(next_population)

# Get best individual
fitnesses = np.array([fitness(ind) for ind in population])
best_idx = fitnesses.argmax()
best_individual = population[best_idx]
selected_features = data.drop(columns=['diagnosis']).columns[best_individual == 1].tolist()

print("\nSelected Features:")
print(selected_features)

# Save new dataset with only selected features + diagnosis column
new_data = data[selected_features + ['diagnosis']]
new_data.to_csv("breast_cancer_selected_features.csv", index=False)
print("\nNew dataset saved as 'breast_cancer_selected_features.csv'")
