from catboost import CatBoostRegressor


def create_catboost_model(params=None):
    """
    Create and return a CatBoostRegressor model with specified parameters.

    Args:
        params (dict, optional): A dictionary of parameters to configure the CatBoostRegressor.
                                 If None, default parameters will be used.

    Returns:
        CatBoostRegressor: An instance of CatBoostRegressor configured with the given parameters.
    """
    if params is None:
        params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'RMSE',
            'verbose': False
        }
    
    model = CatBoostRegressor(**params)
    return model


def train_catboost_model(model, X_train, y_train, X_valid=None, y_valid=None, cat_features=None):
    """
    Train the CatBoostRegressor model on the provided training data.

    Args:
        model (CatBoostRegressor): The CatBoostRegressor model to be trained.
        X_train (array-like): Training feature data.
        y_train (array-like): Training target data.
        X_valid (array-like, optional): Validation feature data. Defaults to None.
        y_valid (array-like, optional): Validation target data. Defaults to None.

    Returns:
        CatBoostRegressor: The trained CatBoostRegressor model.
    """
    if X_valid is not None and y_valid is not None:
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_features, use_best_model=True)
    else:
        model.fit(X_train, y_train, cat_features=cat_features)
    
    return model