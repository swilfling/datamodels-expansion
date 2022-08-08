from datamodels.processing.datascaler import Normalizer
from datamodels.wrappers.expandedmodel import ExpandedModel, TransformerSet
from datamodels import RandomForestRegression, LinearRegression
import numpy as np
from sklearn.preprocessing import FunctionTransformer


def test_instantiate_expandedmodel():

    model = RandomForestRegression(x_scaler_class=Normalizer, name="RF",  parameters={})
    transformer = FunctionTransformer(func=np.sin)
    expanded_model = ExpandedModel(transformers=TransformerSet(transformers=[transformer]), model=model, feature_names=[])
    estimator = expanded_model.model

    transformer_from_model = expanded_model.transformers.get_transformer_by_name('polynomialexpansion')
    assert(isinstance(estimator, RandomForestRegression))
    assert(isinstance(transformer_from_model,FunctionTransformer))
    assert(transformer_from_model == transformer)


def test_expandedmodel_data():
    data = np.random.randn(100,1,3)
    target = np.random.randn(100,1,1)
    model_1 = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    model_2 = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    transformer = FunctionTransformer(func=np.sin)
    expander_set = TransformerSet(transformers=[transformer])
    expanded_model = ExpandedModel(transformers=expander_set, model=model_2, feature_names=[])

    # Check transformation
    data_transformed = transformer.fit_transform(data)
    data_transformed_exp_model = expanded_model.transform_features(data)
    assert(np.all(data_transformed == data_transformed_exp_model))

    # Check training
    model_1.train(data_transformed, target)
    expanded_model.train(data, target)
    coef_1 = model_1.model.coef_
    coef_2 = expanded_model.model.model.coef_
    assert(np.all(np.isclose(coef_1,coef_2)))

    # Check prediction
    x_test = np.random.randn(20,1,3)
    x_test_transformed = transformer.transform(x_test)
    y1 = model_1.predict(x_test_transformed)
    y2 = expanded_model.predict(x_test)
    print(y1.flatten())
    print(y2.flatten())
    print(y1.shape)
    print(y2.shape)
    assert(np.all(np.isclose(y1.flatten(),y2.flatten())))


if __name__ == "__main__":
    test_instantiate_expandedmodel()
    test_instantiate_transformerset_feat_sel()
    test_expandedmodel_data()

