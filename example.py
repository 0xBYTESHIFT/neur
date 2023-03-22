import numpy as np

# Define the input matrix
input_matrix = np.array([[1, 2]])

# Define the weight matrices
weight_matrix_1 = np.array([[0.5, 0.2], 
                            [0.3, 0.4]])

weight_matrix_2 = np.array([[-0.1], [0.7]])

weight_matrix_3 = np.array([[0.2, 0.3, 0.1]])

print( "layer1:", weight_matrix_1, " ", weight_matrix_1.shape )
print( "layer2:", weight_matrix_2, " ", weight_matrix_2.shape )
print( "layer3:", weight_matrix_3, " ", weight_matrix_3.shape )

# Perform matrix multiplication and activation function
hidden_layer = np.dot(input_matrix, weight_matrix_1)
hidden_layer_activated = 1 / (1 + np.exp(-hidden_layer))
print(hidden_layer_activated )

second_layer = np.dot(hidden_layer_activated, weight_matrix_2)
second_layer_activated = 1 / (1 + np.exp(-second_layer))
print(second_layer_activated )

output_layer = np.dot(second_layer_activated, weight_matrix_3)
output_layer_activated = 1 / (1 + np.exp(-output_layer))

# Print the output
print(output_layer_activated)
