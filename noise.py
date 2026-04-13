"""
Quantum Neural Network with Depolarizing Noise

This module implements a QNN with:
1. Simple Parameterized Quantum Circuit (PQC)
2. TorchLayer integration for PyTorch compatibility
3. Default.mixed device for mixed-state simulation
4. Configurable shot numbers
5. Depolarizing noise after every quantum operation
"""

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn


class QuantumNeuralNetwork(nn.Module):
    """
    Quantum Neural Network with depolarizing noise.

    Features:
    - Angle encoding for input data
    - Trainable rotation layers (RX, RY, RZ)
    - Entangling CNOT gates
    - Depolarizing noise after each operation
    - Configurable shots for measurement
    """

    def __init__(self, n_qubits=4, n_layers=2, noise_prob=0.01, n_shots=1000):
        """
        Initialize the Quantum Neural Network.

        Args:
            n_qubits (int): Number of qubits in the circuit
            n_layers (int): Number of parameterized layers
            noise_prob (float): Depolarizing noise probability (0 to 1)
            n_shots (int): Number of measurement shots
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.noise_prob = noise_prob
        self.n_shots = n_shots

        # Create device with default.mixed for noise support
        self.dev = qml.device('default.mixed', wires=n_qubits, shots=n_shots)

        # Define weight shapes for the quantum circuit
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        # Create the quantum circuit and TorchLayer
        self._create_circuit()

    def _create_circuit(self):
        """Create the quantum circuit with depolarizing noise."""

        @qml.qnode(self.dev, interface='torch', diff_method='parameter-shift')
        def qnn_circuit(inputs, weights):
            """
            Parameterized Quantum Circuit with depolarizing noise.

            Args:
                inputs: Input data to encode (shape: [n_qubits])
                weights: Trainable parameters (shape: [n_layers, n_qubits, 3])

            Returns:
                List of expectation values for Pauli-Z on each qubit
            """
            # Encode inputs with angle encoding + depolarizing noise
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.DepolarizingChannel(self.noise_prob, wires=i)

            # Parameterized layers with depolarizing noise
            for layer in range(self.n_layers):
                # Rotation gates with noise
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

                    qml.RZ(weights[layer, i, 2], wires=i)
                    qml.DepolarizingChannel(self.noise_prob, wires=i)

                # Entangling layer with noise
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                    qml.DepolarizingChannel(self.noise_prob, wires=i)
                    qml.DepolarizingChannel(self.noise_prob, wires=i + 1)

                # Connect last to first qubit (ring topology)
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                qml.DepolarizingChannel(self.noise_prob, wires=self.n_qubits - 1)
                qml.DepolarizingChannel(self.noise_prob, wires=0)

            # Measure Pauli-Z expectations on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Store circuit and create TorchLayer
        self.circuit = qnn_circuit
        self.qlayer = qml.qnn.TorchLayer(qnn_circuit, self.weight_shapes)

    def set_shots(self, n_shots):
        """
        Change the number of measurement shots.

        Args:
            n_shots (int): New number of shots
        """
        self.n_shots = n_shots
        self.dev = qml.device('default.mixed', wires=self.n_qubits, shots=n_shots)
        self._create_circuit()
        print(f"Updated shots to {n_shots}")

    def set_noise_prob(self, noise_prob):
        """
        Change the depolarizing noise probability.

        Args:
            noise_prob (float): New noise probability (0 to 1)
        """
        self.noise_prob = noise_prob
        self._create_circuit()
        print(f"Updated noise probability to {noise_prob}")

    def forward(self, x):
        """
        Forward pass through the quantum circuit.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, n_features]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_qubits]
        """
        # Handle input dimension mismatch
        if x.shape[-1] != self.n_qubits:
            if not hasattr(self, 'input_projection'):
                self.input_projection = nn.Linear(x.shape[-1], self.n_qubits).to(x.device)
            x = self.input_projection(x)

        # Apply quantum circuit
        return self.qlayer(x)

    def get_circuit_info(self):
        """Print information about the quantum circuit."""
        print("="*60)
        print("Quantum Neural Network Configuration")
        print("="*60)
        print(f"Device: default.mixed")
        print(f"Qubits: {self.n_qubits}")
        print(f"Layers: {self.n_layers}")
        print(f"Depolarizing noise probability: {self.noise_prob}")
        print(f"Measurement shots: {self.n_shots}")
        print(f"Weight shape: {self.weight_shapes['weights']}")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        print("="*60)

    def visualize_circuit(self):
        """Visualize the quantum circuit."""
        try:
            # Create sample inputs for visualization
            sample_weights = torch.randn(self.n_layers, self.n_qubits, 3)
            sample_input = torch.randn(self.n_qubits)

            # Draw circuit
            fig, ax = qml.draw_mpl(self.circuit)(sample_input, sample_weights)
            print("Circuit visualization generated")
            return fig
        except Exception as e:
            print(f"Could not visualize circuit: {e}")
            # Fallback to text representation
            print(qml.draw(self.circuit)(sample_input, sample_weights))


class QNNClassifier(nn.Module):
    """
    Binary classifier using Quantum Neural Network.

    Combines QNN with a classical linear layer for classification.
    """

    def __init__(self, n_qubits=4, n_layers=2, noise_prob=0.01, n_shots=1000):
        """
        Initialize QNN classifier.

        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of quantum layers
            noise_prob (float): Depolarizing noise probability
            n_shots (int): Number of measurement shots
        """
        super().__init__()
        self.qnn = QuantumNeuralNetwork(n_qubits, n_layers, noise_prob, n_shots)
        self.fc = nn.Linear(n_qubits, 1)

    def forward(self, x):
        """Forward pass through QNN and classifier."""
        x = self.qnn(x)
        x = self.fc(x)
        return x.squeeze()


def demo():
    """Run a demonstration of the QNN with depolarizing noise."""
    print("\n" + "="*60)
    print("QNN with Depolarizing Noise - Demo")
    print("="*60 + "\n")

    # Initialize QNN
    n_qubits = 4
    n_layers = 2
    noise_prob = 0.01
    n_shots = 1000

    qnn = QuantumNeuralNetwork(n_qubits, n_layers, noise_prob, n_shots)
    qnn.get_circuit_info()

    # Test forward pass
    print("\n1. Testing forward pass...")
    batch_size = 3
    sample_input = torch.randn(batch_size, n_qubits)

    print(f"Input shape: {sample_input.shape}")
    print(f"Input data:\n{sample_input}\n")

    output = qnn(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output data:\n{output}\n")

    # Test shot variation
    print("\n2. Testing different shot configurations...")
    print("-" * 60)
    for shots in [100, 500, 1000, 5000]:
        qnn.set_shots(shots)
        output = qnn(sample_input)
        print(f"Shots={shots:5d} | Output mean: {output.mean().item():+.4f} | Output std: {output.std().item():.4f}")

    # Test training on simple task
    print("\n3. Training example on binary classification...")
    print("-" * 60)

    # Create simple dataset
    n_samples = 20
    X_train = torch.randn(n_samples, n_qubits)
    y_train = torch.randint(0, 2, (n_samples,)).float()

    # Create classifier and optimizer
    model = QNNClassifier(n_qubits, n_layers, noise_prob, 1000)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print(f"Training for 5 epochs...")
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == y_train).float().mean()

        print(f"Epoch {epoch+1}/5 | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.4f}")

    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    demo()
