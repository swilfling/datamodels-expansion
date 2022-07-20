import pickle

from . import Model


class LinearModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        from sklearn.dummy import DummyRegressor

        self.model = DummyRegressor()

    def get_coef(self):
        return self.model.coef_

    def reshape(self, arr):
        if arr.shape[1] == arr.shape[2] == 1:
            arr = arr.ravel()
        else:
            arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
        return arr

    def train_model(self, x, y, **kwargs):
        x = self.reshape(x)
        y = self.reshape(y)

        self.model.fit(x, y, **kwargs)

    def predict_model(self, x):
        x = self.reshape(x)
        return self.model.predict(x)

    def save(self, path="data/models/LinearRegression"):
        super(LinearModel, self).save(path)
        with open(f"{path}/model.pickle", "wb") as file:
            pickle.dump(self.model, file)

    def load_model(self, path="data/models/LinearRegression"):
        super(LinearModel, self).load_model(path)
        with open(f"{path}/model.pickle", "rb") as file:
            self.model = pickle.load(file)
