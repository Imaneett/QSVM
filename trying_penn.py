import numpy as np
import penn_treebank
import qiskit
from qiskit import Aer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name


ptb = penn_treebank.load_data()

# Split dataset into training and testing sets
training_size = 500
testing_size = 100
training_data, training_labels = split_dataset_to_data_and_labels(ptb[:training_size])
testing_data, testing_labels = split_dataset_to_data_and_labels(ptb[training_size:training_size+testing_size])

# Initialize quantum map
feature_map = RawFeatureVector(len(training_data[0]))


qsvm = QSVM(feature_map, training_data, training_labels, multiclass_extension=AllPairs())

backend = Aer.get_backend('q_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)
result = qsvm.run(quantum_instance)

# Calculate accuracy of QSVM algorithm
predicted_labels = qsvm.predict(testing_data)
accuracy = np.mean(predicted_labels == testing_labels)

print("Accuracy of QSVM algorithm: {}".format(accuracy))
