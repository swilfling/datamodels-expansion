import numpy as np

import pytest

from datamodels import (
    Model, 
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    PLSRegression,
    WeightedLS,
    SymbolicRegression,
    RuleFitRegression,
    RandomForestRegression,
    SupportVectorRegression,
    XGBoost,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    EncoderDecoderLSTM,
    GRU,
    RecurrentNetwork,
    NeuralNetwork_sklearn
)


@pytest.mark.parametrize('model_class', [
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    PLSRegression,
    RandomForestRegression,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    GRU,
    RecurrentNetwork,
    NeuralNetwork_sklearn
])
def test_predict_sequence_save_and_load(model_class, tmpdir):
    test_model_path = tmpdir.join('testModel')

    model_name = 'testModel'

    lookback = 5
    lookahead = 1

    number_input_features = 3
    number_target_features = 1

    x_train = np.random.random((15, lookback + 1, number_input_features))
    y_train = np.random.random((15, lookahead + 1, number_target_features))
    x_shape = (lookback + 1, number_input_features)
    y_shape = (lookahead + 1, number_target_features)

    model = model_class(name=model_name)
    model.train(x_train, y_train)
    model.save(test_model_path)

    model_from_file = Model.load(test_model_path)

    assert model_from_file.name == model_name
    assert model_from_file.x_shape == x_shape
    assert model_from_file.y_shape == y_shape
    assert model_from_file.x_scaler is not None
    assert model_from_file.y_scaler is not None

    x_test = np.random.random((5, lookback + 1, number_input_features))

    y_pred = model_from_file.predict(x_test)
    assert y_pred.shape == (5, lookahead + 1, number_target_features)

@pytest.mark.parametrize('model_class', [
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    PLSRegression,
    WeightedLS,
    SymbolicRegression,
    RuleFitRegression,
    RandomForestRegression,
    SupportVectorRegression,
    XGBoost,
    NeuralNetwork,
    VanillaLSTM,
    ConvolutionNetwork,
    ConvolutionLSTM,
    EncoderDecoderLSTM,
    GRU,
    RecurrentNetwork,
    NeuralNetwork_sklearn
])
def test_predict_single_value_save_and_load(model_class, tmpdir):
    test_model_path = tmpdir.join('testModel')

    model_name = 'testModel'

    lookback = 5
    lookahead = 1

    number_input_features = 3
    number_target_features = 1

    x_train = np.random.random((15, lookback + 1, number_input_features))
    y_train = np.random.random((15, 1, number_target_features))
    
    x_shape = (lookback + 1, number_input_features)
    y_shape = (1, number_target_features)

    model = model_class(name=model_name)
    model.train(x_train, y_train)
    model.save(test_model_path)

    model_from_file = Model.load(test_model_path)

    assert model_from_file.name == model_name
    assert model_from_file.x_shape == x_shape
    assert model_from_file.y_shape == y_shape
    assert model_from_file.x_scaler is not None
    assert model_from_file.y_scaler is not None

    x_test = np.random.random((5, lookback + 1, number_input_features))

    y_pred = model_from_file.predict(x_test)
    assert y_pred.shape == (5, 1, number_target_features)

@pytest.mark.parametrize('model_class', [
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    PLSRegression,
    RandomForestRegression
])
def test_reshaping_on_multiout_models(model_class):
    batch = 15
    lookback = 5

    number_input_features = 3
    number_target_features = 2

    model = model_class()

    x_train = np.random.random((batch, 1, 1))
    y_train = np.random.random((batch, 1, 1))

    x = model.reshape(x_train)
    y = model.reshape(y_train)

    assert x.shape == (batch,)
    assert y.shape == (batch,)

    x_train = np.random.random((batch, lookback + 1, number_input_features))
    y_train = np.random.random((batch, 1, number_target_features))

    x = model.reshape(x_train)
    y = model.reshape(y_train)

    assert x.shape == (batch, (lookback + 1) * number_input_features)
    assert y.shape == (batch, 1 * number_target_features)

@pytest.mark.parametrize('model_class', [
    SupportVectorRegression,
    WeightedLS,
    SymbolicRegression,
    RuleFitRegression,
    XGBoost,
])
def test_reshaping_on_singleout_models(model_class):
    batch = 15
    lookback = 5

    number_input_features = 3
    number_target_features = 2

    model = model_class()

    x_train = np.random.random((batch, 1, 1))
    y_train = np.random.random((batch, 1, 1))

    x = model.reshape_x(x_train)
    y = model.reshape_y(y_train)

    assert x.shape == (batch,)
    assert y.shape == (batch,)

    x_train = np.random.random((batch, lookback + 1, number_input_features))

    x = model.reshape_x(x_train)

    assert x.shape == (batch, (lookback + 1) * number_input_features)

    y_train = np.random.random((batch, 1, number_target_features))

    with pytest.raises(Exception):
        y = model.reshape_y(y_train)