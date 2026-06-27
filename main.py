import pennylane as qml
from pennylane import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD AND DOWNSCALE MNIST IMAGES
# ==========================================
print("Loading and preparing downscaled MNIST data...")
# We use scikit-learn's built-in 8x8 digit dataset as a base
digits = load_digits()
X_raw = digits.images
y_raw = digits.target

# Filter for binary classification: only keep digits '0' and '1'
binary_filter = (y_raw == 0) | (y_raw == 1)
X_filtered = X_raw[binary_filter]
y_filtered = y_raw[binary_filter]

# To make it truly beginner-quantum friendly, we downsample 8x8 images to 2x2 pixels
X_2x2 = []
for img in X_filtered:
    # Basic pixel pooling: average 4x4 blocks to get a 2x2 image
    top_left = np.mean(img[0:4, 0:4])
    top_right = np.mean(img[0:4, 4:8])
    bottom_left = np.mean(img[4:8, 0:4])
    bottom_right = np.mean(img[4:8, 4:8])
    X_2x2.append([top_left, top_right, bottom_left, bottom_right])

X_2x2 = np.array(X_2x2)

# Normalize pixel values between [0, pi] so they act as clean rotation angles
scaler = MinMaxScaler(feature_range=(0, np.pi))
X_normalized = scaler.fit_transform(X_2x2)

# Convert labels from {0, 1} to {-1, 1} for quantum measurement interpretation
y_labels = np.where(y_filtered == 0, -1, 1)

# Split into 80% Train and 20% Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

# ==========================================
# 2. SETUP THE QUANTUM NEURAL NETWORK
# ==========================================
# 4 pixels = 4 qubits needed
num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

# A single programmable layer of our Quantum Neural Network
def qnn_layer(weights, wires):
    for i in range(num_qubits):
        qml.RY(weights[i], wires=wires[i])
    # Entangle neighboring qubits to recognize patterns across pixels
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[wires[i], wires[i+1]])
    qml.CNOT(wires=[wires[num_qubits-1], wires[0]])

@qml.qnode(dev)
def quantum_circuit(weights, x):
    # Step A: Encode the 4 pixels into the 4 qubits using Angle Embedding
    qml.AngleEmbedding(x, wires=range(num_qubits), rotation='X')
    
    # Step B: Apply multiple trainable variational quantum layers
    for layer_weights in weights:
        qnn_layer(layer_weights, wires=range(num_qubits))
        
    # Step C: Read out the expectation value of the first qubit
    return qml.expval(qml.PauliZ(0))

# ==========================================
# 3. DEFINE LOSS & ACCURACY FUNCTIONS
# ==========================================
def quantum_classifier(weights, bias, x):
    return quantum_circuit(weights, x) + bias

def squared_loss(weights, bias, X, y):
    loss = 0
    for x, target in zip(X, y):
        prediction = quantum_classifier(weights, bias, x)
        loss += (prediction - target) ** 2
    return loss / len(X)

def calculate_accuracy(weights, bias, X, y):
    predictions = [np.sign(quantum_classifier(weights, bias, x)) for x in X]
    return np.mean(np.array(predictions) == y)

# ==========================================
# 4. INITIALIZE PARAMETERS & TRAIN
# ==========================================
num_layers = 2
np.random.seed(42)
# Initialize weights randomly for 2 layers, 4 qubits each
weights_init = 0.1 * np.random.randn(num_layers, num_qubits, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

opt = qml.GradientDescentOptimizer(stepsize=0.2)
batch_size = 10

weights = weights_init
bias = bias_init

print("\nStarting Quantum Training Loop...")
print("---------------------------------")

for epoch in range(10):
    # Mini-batch sampling for faster simulator performance
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_batch = X_train[batch_index]
    y_batch = y_train[batch_index]
    
    # Update the quantum weights and classical bias terms via Gradient Descent
    weights, bias = opt.step(lambda w, b: squared_loss(w, b, X_batch, y_batch), weights, bias)
    
    # Evaluate performance on the training set
    train_acc = calculate_accuracy(weights, bias, X_train, y_train)
    current_loss = squared_loss(weights, bias, X_train, y_train)
    
    print(f"Epoch {epoch+1:2d} | Loss: {current_loss:.4f} | Train Accuracy: {train_acc*100:.1f}%")

# ==========================================
# 5. FINAL TEST EVALUATION
# ==========================================
test_accuracy = calculate_accuracy(weights, bias, X_test, y_test)
print("---------------------------------")
print(f"Final Test Accuracy on Unknown Digits: {test_accuracy*100:.1f}%")
