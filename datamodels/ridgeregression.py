from . import LinearModel


class RidgeRegression(LinearModel):

    def __init__(self, parameters=None, **kwargs):
        super().__init__(**kwargs)

        if parameters is None:
            parameters = {'alpha':0.5}
        self.sample_weight = kwargs.get('sample_weight', None)

        from sklearn.linear_model import Ridge
        self.model = Ridge(**parameters)

    def train_model(self, x, y, **kwargs):
        super(RidgeRegression, self).train_model(x, y, sample_weight=self.sample_weight, **kwargs)