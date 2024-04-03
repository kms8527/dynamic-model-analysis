import onnx

# Load the ONNX model
model = onnx.load('/home/a/Carmaker_11_1/ros/ros1_ws/src/random_forest.onnx')

# Get input and output node names
input_node_names = [node.name for node in model.graph.input]
output_node_names = [node.name for node in model.graph.output]

print('Input node names:', input_node_names)
print('Output node names:', output_node_names)