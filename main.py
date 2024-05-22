from numpy import exp, array, random, dot

class NN:
    def __init__(self):
        random.seed(1)
        self.weights = 2 * random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustment

    def think(self, inputs):
        return self.sigmoid(dot(inputs, self.weights))

if __name__ == "__main__":
    nn = NN()
    print("Random starting weights: ")
    print(nn.weights)

    inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    outputs = array([[0, 1, 1, 0]]).T

    nn.train(inputs, outputs, 10000)
    print("New weights after training: ")
    print(nn.weights)

    print("Considering new situation [1, 0, 0] ->?: ")
    print(nn.think(array([1, 0, 0])))