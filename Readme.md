# Wine Classification using Genetic Algorithms  and Holographic Reduced Representation (HRR)

This script employs a Genetic Algorithm-based Neural Network and Holographic Reduced Representation (HRR) to classify wine types based on the UCI ML Wine recognition dataset. The Neural Network is implemented using the Keras library.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Script Details](#script-details)
4. [Outputs](#outputs)
5. [Acknowledgements](#acknowledgements)

## Installation
This script uses the following Python libraries:

- numpy
- keras
- sklearn
- random

You can install them using `pip`:

```sh
pip install numpy keras sklearn
```

## Usage

Simply run the Python script in a suitable Python environment:

```sh
python main.py
```

The script will  also load an existing model and HRR weights, if available, for inmediate use. If you wish to train a new model, simply delete the existing files.
## Script Details

This script consists of several steps:

1. **Loading the wine dataset**: The UCI ML Wine recognition dataset is loaded using the `sklearn.datasets.load_wine` function.

2. **Data Preprocessing**: The loaded dataset is split into training and testing datasets. The features are then standardized using the `StandardScaler` class from `sklearn.preprocessing`.

3. **Holographic Reduced Representation (HRR) Encoding**: A function `encode_hrr` is provided to apply HRR encoding to the features.

4. **Genetic Algorithm**: This algorithm includes creating individuals, initializing the population, defining the fitness function, performing selection, crossover, and mutation operations.

5. **Prediction**: The trained model predicts the wine class based on user input.

## Outputs

- **Model**: The best individual model is saved as 'model.h5'.
- **Model Weights**: The weights of the best individual model are saved as 'model_weights.h5'.
- **HRR Weights**: The HRR weights of the best individual are saved as 'hrr_weights.npy'.
- **Accuracy Score**: The final accuracy score of the best individual on the test set is printed to the console.
- **Confusion Matrix and Classification Report**: The confusion matrix and the classification report are also printed to the console, providing detailed insights into the performance of the model.

## Acknowledgements

This script is based on the [UCI ML Wine recognition dataset](https://archive.ics.uci.edu/ml/datasets/wine).
