import numpy as np
import csv
import sys
'''
README: Call the code by calling python NNpy x y z train.csv test.csv

x is the depth value, y is the breadth value, and z is the number of training cycles.
You can replace the csv files with different files. I chose to use the sigmoid activation
function for all layers of the NN, so my accuracy usually rounds out to be 56.69% with my smaller
data set and 50.74% with the larger set.
If no output had been printed, the program is likely running slower due to the large file
size or the large number of training cycles. 
'''

class FeedForwardNN:
  def __init__(self, depth, breadth, input_size):
    self.depth = depth
    self.breadth = breadth
    self.weights = []
    self.biases = []
    
    # Initialize weights and biases
    for i in range(depth):
      in_size = input_size if i == 0 else breadth
      out_size = 1 if i == depth - 1 else breadth
      self.weights.append(np.random.randn(in_size, out_size) * 0.01)
      self.biases.append(np.zeros((1, out_size)))

  #Sigmoid function from Lec 21
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))
  
  def sigmoid_derivative(self, x):
    return x * (1 - x)
  
  #Forward alg that thats the previous activation, accounts for 
  #weight and bias, and then calculates the new value that is output
  #by the activation functiom I chose
  def forward(self, X):
    activations = [X]
    for i in range(self.depth):
      z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
      activation = self.sigmoid(z)
      activations.append(activation)
    return activations

  #Backward propogation using Squared Loss L2. 
  def backward(self, activations, true_y_val, learning_rate):
    m = true_y_val.shape[0]
    # Derivative of squared loss
    delta = activations[-1] - true_y_val  
    for i in reversed(range(self.depth)):
      dz = delta * self.sigmoid_derivative(activations[i + 1])
      dw = np.dot(activations[i].T, dz) / m
      #Summing w zero to be able to do arithmetic with dz
      db = np.sum(dz, 0) / m
      delta = np.dot(dz, self.weights[i].T) 

      # Update weights and biases
      self.weights[i] -= learning_rate * dw
      self.biases[i] -= learning_rate * db

  #Trains the model based on the number of input cycles. 
  def train(self, X, y, cycles, learning_rate):
    for i in range(cycles):
      activations = self.forward(X)
      self.backward(activations, y, learning_rate)

  #Uses the forward algorithm to make predictions
  def predict(self, X):
    activations = self.forward(X)
    return (activations[-1] > 0.5).astype(int)

#Loads the data from each file, assuming there are no blank lines in the file.
#Stores the 
def load_data(filepath):
  print(f"Loading {filepath}...")
  data = []
  with open(filepath, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
      data.append([float(value) for value in row])

  data = np.array(data)
  #X contains the Normalized pixel values and y contains the first value from each line of data (class)
  X = data[:, 1:] / 255.0
  y = data[:, 0].reshape(-1, 1)
    
  print(f"Finished loading {filepath}.")
  return X, y
  
def main():
  if len(sys.argv) != 6:
    print("Usage: python NN.py <D> <B> <C> <Trainfile> <Testfile>")
    return

  # Parse arguments
  D = int(sys.argv[1])
  B = int(sys.argv[2])
  C = int(sys.argv[3])
  train_file = sys.argv[4]
  test_file = sys.argv[5]

  # Load training and testing data
  train_X, train_Y = load_data(train_file)
  test_X, test_Y = load_data(test_file)

  input_size = train_X.shape[1]
  nn = FeedForwardNN(D, breadth = B, input_size = input_size)

  # Train the network
  print("Training...")
  nn.train(train_X, train_Y, cycles = C, learning_rate = 0.1)

  # Test the network
  print("Testing...")
  predictions = nn.predict(test_X)
  accuracy = np.mean(predictions == test_Y)
  print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
  main()
