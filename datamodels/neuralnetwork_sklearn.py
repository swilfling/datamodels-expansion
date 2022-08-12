from sklearn.neural_network import MLPRegressor
from . import Model


class NeuralNetwork_sklearn(Model):
    """
    the NeuralNetwork class acts as a wrapper for networks.
    the neural network can be customized by passing a custom function for any of the three parameters in the constructor.
    if nothing else is passed to the constructor the class uses the implementations above to build, compile and train the model.

    DO NOT CHANGE the functions here if you want a different network, 
    instead implement any of the three functions and pass them to the constructor in the file where you use the network.

    """

    def __init__(self, n_hidden_layers=2, n_hidden_neurons=10, **kwargs):
        super().__init__(**kwargs)
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_neurons = n_hidden_neurons
        hidden_layer_sizes = [self.n_hidden_neurons for n in range(self.n_hidden_layers)]
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes)

    def reshape(self, X):
        if X.ndim == 3:
            return X.reshape(X.shape[0],X.shape[2])
        return X

    def train_model(self, x, y, **kwargs):
        x = self.reshape(x)
        y = self.reshape(y)
        self.model.fit(x, y)

    def predict_model(self, x):
        return self.model.predict(x)

    def save(self, path="data/models/NeuralNetwork_sklearn"):
        super().save(path)

    def load_model(self, path="data/models/NeuralNetwork_sklearn"):
        super().save(path)
