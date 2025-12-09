"""
Neuroevolution: Evolve ANN architecture + weights with a Genetic Algorithm
Dataset: car-vgood.txt (ARFF-like). Place file in same directory or update DATA_PATH.
Notes:
 - Fitness = validation F1-score (macro). Suitable for imbalanced classes.
 - Genotype: [arch_genes | weight_genes]
    arch_genes: num_layers (1..MAX_HIDDEN), neurons per layer (1..MAX_NEURON) for each hidden slot,
                activation id per layer (0:relu,1:tanh,2:sigmoid)
    weight_genes: fixed-length real vector that is sliced depending on decoded architecture
"""

import numpy as np
import pandas as pd
import random
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import json

# ---------------------------
# Configurable hyperparameters
# ---------------------------
DATA_PATH = "car-vgood.dat"
RANDOM_SEED = 1

POP_SIZE = 120
N_GENERATIONS = 200
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9

# Mutation probabilities
MUTATE_ARCH_PROB = 0.20  # mutate an architecture gene
MUTATE_WEIGHT_PROB = 0.15  # per-weight mutation probability
WEIGHT_MUTATION_SCALE = 0.5

# Architecture limits (design choice)
MAX_HIDDEN = 4  # maximum hidden layers we allow in genotype
MAX_NEURONS = 200  # maximum neurons per hidden layer slot
INPUT_BIAS = True

# Activation mapping
ACTIVATIONS = ["relu", "tanh", "sigmoid"]

# Fitness uses validation F1 (macro)
USE_F1 = True

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------
# Utility: Load ARFF-like file
# ---------------------------
def load_arff_like_txt(path: str):
    lines = Path(path).read_text().splitlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("@data"):
            data_start = i + 1
            break
    rows = []
    for ln in lines[data_start:]:
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        parts = ln.split(",")
        if len(parts) == 7:
            rows.append(parts)
    cols = ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety", "Class"]
    df = pd.DataFrame(rows, columns=cols)
    return df


# ---------------------------
# Load and preprocess dataset
# ---------------------------
df = load_arff_like_txt(DATA_PATH)
print("Loaded rows:", len(df))
print("Class counts:\n", df["Class"].value_counts())

X_raw = df.iloc[:, :-1]
y_raw = df["Class"].map({"positive": 1, "negative": 0}).astype(int).values

# One-hot encode categorical inputs
ohe = OneHotEncoder(handle_unknown="ignore")
X_enc = ohe.fit_transform(X_raw).toarray()

# Add bias to inputs? We'll keep bias as explicit in weight decoding
n_inputs = X_enc.shape[1]
print("Input features after OHE:", n_inputs)

# Make full dataset arrays
X = X_enc
y = y_raw

# Train/validation/test split (stratified)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    random_state=RANDOM_SEED + 1,
    stratify=y_train_full,
)

print("Train:", X_train.shape[0], "Val:", X_val.shape[0], "Test:", X_test.shape[0])


# ---------------------------
# Neural network forward pass (numpy)
# ---------------------------
def activation_forward(z, act_name):
    if act_name == "relu":
        return np.maximum(0, z)
    elif act_name == "tanh":
        return np.tanh(z)
    elif act_name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError("Unknown activation")


def activation_deriv(a, act_name):
    # Not needed for GA, but kept for completeness
    if act_name == "relu":
        return (a > 0).astype(float)
    elif act_name == "tanh":
        return 1 - a**2
    elif act_name == "sigmoid":
        return a * (1 - a)
    else:
        raise ValueError("Unknown activation")


# ---------------------------
# Genotype encoding / decoding
# ---------------------------
# Architecture genes layout:
#  - gene 0: num_hidden_layers (1..MAX_HIDDEN)
#  - for i in 0..MAX_HIDDEN-1:
#       gene: neurons_i (1..MAX_NEURONS)
#       gene: act_i (0..len(ACTIVATIONS)-1)


# Weight genes: we allocate enough weights for the largest possible network:
# Input -> hidden1 -> hidden2 -> hidden3 -> output
# For each layer we need (in_units * out_units) + out_units (bias)
def compute_max_weight_length(n_inputs, max_hidden, max_neurons):
    lengths = []
    prev = n_inputs
    for _ in range(max_hidden):
        lengths.append(prev * max_neurons + max_neurons)  # weights + biases
        prev = max_neurons
    # final output layer (to one output neuron)
    lengths.append(prev * 1 + 1)
    return sum(lengths), lengths


MAX_WEIGHT_LEN, per_layer_weight_len = compute_max_weight_length(
    n_inputs, MAX_HIDDEN, MAX_NEURONS
)
print("Allocated weight gene length (max):", MAX_WEIGHT_LEN)


# Helper to pack/unpack genotype
def random_arch_genes():
    num_layers = random.randint(1, MAX_HIDDEN)
    neurons = [random.randint(1, MAX_NEURONS) for _ in range(MAX_HIDDEN)]
    acts = [random.randint(0, len(ACTIVATIONS) - 1) for _ in range(MAX_HIDDEN)]
    # We store all MAX_HIDDEN neurons/acts but only first num_layers used
    arch = {"num_layers": num_layers, "neurons": neurons, "acts": acts}
    return arch


def random_individual():
    arch = random_arch_genes()
    weights = np.random.normal(0, 0.5, size=MAX_WEIGHT_LEN)
    return {"arch": arch, "weights": weights}


def decode_arch(arch):
    num_layers = arch["num_layers"]
    neurons = arch["neurons"][:num_layers]
    acts = [ACTIVATIONS[a] for a in arch["acts"][:num_layers]]
    return num_layers, neurons, acts


def slice_weights_for_arch(weights, arch):
    """
    Given flat weight vector (MAX_WEIGHT_LEN), produce a list of (W,b) for each layer including output.
    W: shape (in_units, out_units), b: shape (out_units,)
    """
    num_layers, neurons, acts = decode_arch(arch)
    layers = []
    idx = 0
    in_units = n_inputs
    # hidden layers
    for layer_i in range(num_layers):
        out_units = neurons[layer_i]
        need = in_units * out_units + out_units
        seg = weights[idx : idx + need]
        W = seg[: in_units * out_units].reshape((in_units, out_units))
        b = seg[in_units * out_units : in_units * out_units + out_units]
        layers.append((W, b, acts[layer_i]))
        idx += need
        in_units = out_units
    # output layer (single neuron)
    out_units = 1
    need = in_units * out_units + out_units
    seg = weights[idx : idx + need]
    W = seg[: in_units * out_units].reshape((in_units, out_units))
    b = seg[in_units * out_units : in_units * out_units + out_units]
    layers.append((W, b, "sigmoid"))  # final activation sigmoid for binary
    return layers


def forward_with_genotype(weights, arch, X_batch):
    """Do forward pass using genotype-decoded weights. Returns probabilities."""
    layers = slice_weights_for_arch(weights, arch)
    a = X_batch.copy()
    for i, (W, b, act) in enumerate(layers):
        z = a.dot(W) + b  # shape (n_samples, out_units)
        a = activation_forward(z, act)
    # a shape (n_samples,1); return flattened probabilities
    probs = a.ravel()
    # ensure in (0,1)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return probs


# ---------------------------
# Fitness function
# ---------------------------


def fitness_of_ind(individual):
    # evaluate on validation set
    probs = forward_with_genotype(individual["weights"], individual["arch"], X_val)
    preds = (probs >= 0.5).astype(int)
    if USE_F1:
        f1 = f1_score(y_val, preds, average="macro")
        return f1
    else:
        return accuracy_score(y_val, preds)


# ---------------------------
# GA operators
# ---------------------------
def tournament_selection(pop, k=TOURNAMENT_K):
    ids = random.sample(range(len(pop)), k)
    best = max(ids, key=lambda i: pop[i]["fitness"])
    # return deep copy to avoid accidental in-place editing
    return {
        "arch": {
            "num_layers": pop[best]["arch"]["num_layers"],
            "neurons": pop[best]["arch"]["neurons"][:],
            "acts": pop[best]["arch"]["acts"][:],
        },
        "weights": pop[best]["weights"].copy(),
        "fitness": pop[best]["fitness"],
    }


def crossover(parent1, parent2):
    """Crossover both arch genes and weight genes.
    Arch crossover: swap some layer slots or mix num_layers.
    Weight crossover: uniform / blend crossover on continuous vector.
    """
    # Crossover architecture (small probability of mixing)
    child_arch1 = {
        "num_layers": parent1["arch"]["num_layers"],
        "neurons": parent1["arch"]["neurons"][:],
        "acts": parent1["arch"]["acts"][:],
    }
    child_arch2 = {
        "num_layers": parent2["arch"]["num_layers"],
        "neurons": parent2["arch"]["neurons"][:],
        "acts": parent2["arch"]["acts"][:],
    }
    if random.random() < 0.5:
        # swap a random hidden-slot (neurons+act) between parents
        slot = random.randrange(0, MAX_HIDDEN)
        child_arch1["neurons"][slot], child_arch2["neurons"][slot] = (
            child_arch2["neurons"][slot],
            child_arch1["neurons"][slot],
        )
        child_arch1["acts"][slot], child_arch2["acts"][slot] = (
            child_arch2["acts"][slot],
            child_arch1["acts"][slot],
        )
    # mix num_layers with 50% chance
    if random.random() < 0.3:
        child_arch1["num_layers"] = parent2["arch"]["num_layers"]
    if random.random() < 0.3:
        child_arch2["num_layers"] = parent1["arch"]["num_layers"]

    # Weight crossover: blend / arithmetic crossover
    w1 = parent1["weights"]
    w2 = parent2["weights"]
    if random.random() < CROSSOVER_RATE:
        alpha = np.random.uniform(0, 1, size=w1.shape)
        c1w = alpha * w1 + (1 - alpha) * w2
        c2w = alpha * w2 + (1 - alpha) * w1
    else:
        c1w = w1.copy()
        c2w = w2.copy()

    child1 = {"arch": child_arch1, "weights": c1w}
    child2 = {"arch": child_arch2, "weights": c2w}
    return child1, child2


def mutate(ind):
    # mutate architecture
    if random.random() < MUTATE_ARCH_PROB:
        # mutate num_layers to neighboring value
        nl = ind["arch"]["num_layers"]
        if random.random() < 0.5:
            nl = max(1, nl - 1)
        else:
            nl = min(MAX_HIDDEN, nl + 1)
        ind["arch"]["num_layers"] = nl
    # mutate neurons or activations in one random slot
    if random.random() < MUTATE_ARCH_PROB:
        slot = random.randrange(0, MAX_HIDDEN)
        # either change neurons or activation
        if random.random() < 0.6:
            # small perturbation
            delta = random.randint(-4, 4)
            new_n = int(np.clip(ind["arch"]["neurons"][slot] + delta, 1, MAX_NEURONS))
            ind["arch"]["neurons"][slot] = new_n
        else:
            # flip activation
            ind["arch"]["acts"][slot] = random.randint(0, len(ACTIVATIONS) - 1)

    # mutate weights: gaussian noise per gene
    mask = np.random.rand(len(ind["weights"])) < MUTATE_WEIGHT_PROB
    gauss = np.random.normal(0, WEIGHT_MUTATION_SCALE, size=ind["weights"].shape)
    ind["weights"][mask] += gauss[mask]

    return ind


# ---------------------------
# Initialize population
# ---------------------------
population = [random_individual() for _ in range(POP_SIZE)]
# Evaluate initial fitness
for ind in population:
    ind["fitness"] = fitness_of_ind(ind)

# ---------------------------
# GA main loop
# ---------------------------
best_history = []
mean_history = []
best_overall = None

for gen in range(N_GENERATIONS):
    # Elitism: keep top 2
    population.sort(key=lambda x: x["fitness"], reverse=True)
    new_pop = [
        {
            "arch": {
                "num_layers": population[0]["arch"]["num_layers"],
                "neurons": population[0]["arch"]["neurons"][:],
                "acts": population[0]["arch"]["acts"][:],
            },
            "weights": population[0]["weights"].copy(),
            "fitness": population[0]["fitness"],
        }
    ]
    new_pop.append(
        {
            "arch": {
                "num_layers": population[1]["arch"]["num_layers"],
                "neurons": population[1]["arch"]["neurons"][:],
                "acts": population[1]["arch"]["acts"][:],
            },
            "weights": population[1]["weights"].copy(),
            "fitness": population[1]["fitness"],
        }
    )

    # Create rest of new population
    while len(new_pop) < POP_SIZE:
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1)
        child2 = mutate(child2)
        # evaluate fitness lazily after full population ready
        new_pop.append(child1)
        if len(new_pop) < POP_SIZE:
            new_pop.append(child2)

    # Evaluate new generation fitness
    for ind in new_pop[2:]:  # first two already have fitness from elites
        ind["fitness"] = fitness_of_ind(ind)

    population = new_pop

    # Logging
    fitness_vals = [ind["fitness"] for ind in population]
    gen_best = max(fitness_vals)
    gen_mean = float(np.mean(fitness_vals))
    best_history.append(gen_best)
    mean_history.append(gen_mean)

    # best overall
    best_ind = max(population, key=lambda x: x["fitness"])
    if best_overall is None or best_ind["fitness"] > best_overall["fitness"]:
        # copy best
        best_overall = {
            "arch": {
                "num_layers": best_ind["arch"]["num_layers"],
                "neurons": best_ind["arch"]["neurons"][:],
                "acts": best_ind["arch"]["acts"][:],
            },
            "weights": best_ind["weights"].copy(),
            "fitness": best_ind["fitness"],
        }

    if (gen + 1) % 10 == 0 or gen == 0:
        print(
            f"Gen {gen + 1:3d} | Best val fitness: {gen_best:.4f} | Mean: {gen_mean:.4f} | Overall best: {best_overall['fitness']:.4f}"
        )

# ---------------------------
# Final evaluation on test set
# ---------------------------
final = best_overall
print("\nBest validation fitness (final):", final["fitness"])
print("Decoded architecture:")
n_layers, neurons, acts = decode_arch(final["arch"])
for i in range(n_layers):
    print(f" Hidden layer {i + 1}: neurons={neurons[i]}, activation={acts[i]}")
print("Output: 1 neuron (sigmoid)")

# Test predictions
probs_test = forward_with_genotype(final["weights"], final["arch"], X_test)
preds_test = (probs_test >= 0.5).astype(int)

print("\nTest accuracy:", accuracy_score(y_test, preds_test))
print("Test F1 (macro):", f1_score(y_test, preds_test, average="macro"))
print(
    "\nClassification report (test):\n",
    classification_report(y_test, preds_test, target_names=["negative", "positive"]),
)
print("Confusion matrix (test):\n", confusion_matrix(y_test, preds_test))

# ---------------------------
# Plots: fitness progression
# ---------------------------
plt.figure(figsize=(8, 4))
plt.plot(best_history, label="best_val_fitness")
plt.plot(mean_history, label="mean_val_fitness")
plt.xlabel("Generation")
plt.ylabel("Validation Fitness (F1 macro)")
plt.title("GA Neuroevolution Progress")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# Optional: Save best genotype to disk (numpy)
# ---------------------------

save_obj = {
    "arch": {
        "num_layers": final["arch"]["num_layers"],
        "neurons": final["arch"]["neurons"],
        "acts": final["arch"]["acts"],
    },
    "weights": final["weights"].tolist(),
    "ohe_categories": [list(cat) for cat in ohe.categories_],
    "input_feature_count": n_inputs,
}
Path("best_genotype.json").write_text(json.dumps(save_obj))
print("\nSaved best_genotype.json (weights + arch + encoder categories).")
