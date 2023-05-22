#This is the main file for the project. It contains the code for the genetic algorithm and the HRR encoding function.
#It also contains the code for the user input and the prediction of the wine class.

#Script explanation:
#1. The genetic algorithm is used to find the best model for the wine dataset.
#2. The HRR encoding function is used to encode the features of the wine dataset.
#3. The user can input the values of the features and the script will predict the wine class.
#4. The script will save the best model in the models folder.
#5. The script will save the HRR weights in the weights folder.

#Advatanges of using HRR encoding:
#1. It is a very simple and efficient way to encode the features.
#2. It is an attractive alternative to one-hot encoding because it does not increase the dimensionality of the data.
#3. It is a good way to encode features that are not ordinal.
#(C)Jesús Vilela Jato, 2021

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
from sklearn.metrics import classification_report
import os


# Set a random seed for reproducibility
np.random.seed(42)

# Load the wine dataset
def load_dataset():
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Preprocess the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the target vector to one-hot encoding
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, scaler


# HRR Encoding function
def encode_hrr(features, weights):
    encoded = np.dot(features, weights)
    return encoded


# Initialize the population
def create_individual():
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=13))  # Set input_dim to 13
    model.add(
        Dense(3, activation='softmax'))  # Set the number of output neurons to 3 and use the softmax activation function
    model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy',
                  metrics=['accuracy'])  # Use categorical_crossentropy loss
    hrr_weights = np.random.randn(13, 13)  # Set the shape of HRR weights to match the number of features
    return {'model': model, 'hrr_weights': hrr_weights}


def init_population(size):
    return [create_individual() for _ in range(size)]


# Fitness function
def fitness(individual, X, y):
    encoded_features = encode_hrr(X, individual['hrr_weights'])
    loss, accuracy = individual['model'].evaluate(encoded_features, y, verbose=0)
    return accuracy


# Tournament selection
def selection(population, X, y, k=3):
    selected = random.sample(population, k)
    fitnesses = [fitness(ind, X, y) for ind in selected]
    return selected[np.argmax(fitnesses)]


# Crossover and mutation
def crossover_and_mutation(parent1, parent2, mutation_rate=0.1):
    child = create_individual()

    # Crossover
    for i, layer in enumerate(child['model'].layers):
        weights = [np.where(np.random.rand(*w1.shape) < 0.5, w1, w2) for w1, w2 in
                   zip(parent1['model'].layers[i].get_weights(), parent2['model'].layers[i].get_weights())]
        layer.set_weights(weights)

    # HRR crossover
    child['hrr_weights'] = np.where(np.random.rand(*parent1['hrr_weights'].shape) < 0.5, parent1['hrr_weights'],
                                    parent2['hrr_weights'])

    # Mutation
    for layer in child['model'].layers:
        weights = layer.get_weights()
        for w in weights:
            mutation_mask = np.random.rand(*w.shape) < mutation_rate
            w += np.random.normal(0, 0.1, w.shape) * mutation_mask
        layer.set_weights(weights)

    # HRR mutation
    mutation_mask = np.random.rand(*child['hrr_weights'].shape) < mutation_rate
    child['hrr_weights'] += np.random.normal(0, 0.1, child['hrr_weights'].shape) * mutation_mask

    return child


def genetic_algorithm(X_train, y_train, X_test, y_test, pop_size=50, num_generations=50):
    population = init_population(pop_size)
    best_fitness = -np.inf  # Initialize the best fitness to a very low number
    for generation in range(num_generations):
        print(f"Generation {generation + 1}")

        # Evaluate population
        fitnesses = [fitness(ind, X_train, y_train) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_individual = population[best_idx]
        current_fitness = fitness(best_individual, X_test, y_test)

        # Update the best individual if the current individual is better
        if current_fitness > best_fitness:
            best_individual = population[best_idx]
            best_fitness = current_fitness

        print(f"Best fitness: {best_fitness}")

        # Create a new population using selection, crossover, and mutation
        new_population = []
        for _ in range(pop_size):
            parent1 = selection(population, X_train, y_train)
            parent2 = selection(population, X_train, y_train)
            child = crossover_and_mutation(parent1, parent2)
            new_population.append(child)

        population = new_population

    return best_individual

def get_user_input():
    feature_names = wine_data.feature_names
    default_values = [round(x, 2) for x in wine_data.data.mean(axis=0)]

    user_features = []
    for feature_name, default_value in zip(feature_names, default_values):
        user_input = float(input(f"Enter the value for {feature_name} (suggested: {default_value}): "))
        user_features.append(user_input)

    return user_features


def predict_wine_class(model, hrr_weights, scaler, user_features):
    # Scale the features using the same scaler used during training
    user_features_scaled = scaler.transform([user_features])

    # Encode the features with HRR weights
    encoded_features = encode_hrr(user_features_scaled, hrr_weights)

    # Predict the wine class
    prediction = model.predict(encoded_features)

    # Return the predicted class
    return np.argmax(prediction)


MODEL_PATH = 'model.h5'
MODEL_WEIGHTS_PATH = 'model_weights.h5'
HRR_WEIGHTS_PATH = 'hrr_weights.npy'

X_train, X_test, y_train, y_test, scaler = load_dataset()

# Check if the model and weight files exist
if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_WEIGHTS_PATH) and os.path.exists(HRR_WEIGHTS_PATH):
    # Load the existing model and HRR weights
    from keras.models import load_model

    best_individual = {}
    best_individual['model'] = load_model(MODEL_PATH)
    best_individual['model'].load_weights(MODEL_WEIGHTS_PATH)
    best_individual['hrr_weights'] = np.load(HRR_WEIGHTS_PATH)

    print("Loaded the existing model and HRR weights from disk.")

    encoded_features = encode_hrr(X_test, best_individual['hrr_weights'])
    loss, accuracy = best_individual['model'].evaluate(encoded_features, y_test, verbose=0)

    print(f"Accuracy on test set: {accuracy}")
else:
    # No existing model found, training a new one
    print("No existing model found. Training a new model...")

    best_individual = genetic_algorithm(X_train, y_train, X_test, y_test)
    encoded_features = encode_hrr(X_test, best_individual['hrr_weights'])
    loss, accuracy = best_individual['model'].evaluate(encoded_features, y_test, verbose=0)

    # Save the entire model and HRR weights
    best_individual['model'].save(MODEL_PATH)
    best_individual['model'].save_weights(MODEL_WEIGHTS_PATH)
    np.save(HRR_WEIGHTS_PATH, best_individual['hrr_weights'])

    print(f"Final accuracy on test set: {accuracy}")

# Load the wine dataset and get the target names
wine_data = datasets.load_wine()
target_names = wine_data.target_names

user_features = get_user_input()
predicted_class = predict_wine_class(best_individual['model'], best_individual['hrr_weights'], scaler, user_features)

predicted_text = target_names[predicted_class]
print(f"The predicted wine class is: {predicted_text}")

# Get predicted values
y_test_class = np.argmax(y_test, axis=1)
y_pred = np.argmax(best_individual['model'].predict(encode_hrr(X_test, best_individual['hrr_weights'])), axis=-1)

scores = metrics.accuracy_score(y_test_class, y_pred)
print('Accuracy: ','{:2.2%}'.format(scores))

cm = metrics.confusion_matrix(y_test_class, y_pred)
print(cm)

print(classification_report(y_test_class, y_pred))

print(np.sum(np.diag(cm)/np.sum(cm)))