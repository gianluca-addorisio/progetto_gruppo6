from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV

def hyperparameter_tune_fs(pipeline, X, y, num_iter=20):
    method_selected = pipeline.named_steps['selector'].fs_method

    params_to_tune = {
        'selector__max_features_to_hold': randint(10, 40)
    }

    #andiamo a definire il range di threshold in base al metodo di fs selezionato
    if method_selected in ['rf', 'xgb', 'ctb', 'mu']:
        #metodi il quale rating è compreso tra 0 e 1
        params_to_tune['selector__threshold'] = uniform(0,0.02)
    elif method_selected == 'corr_matrix':
        params_to_tune['selector__threshold'] = uniform(0,0.5)
    else:
        params_to_tune['selector__threshold'] = uniform(0)
    search = RandomizedSearchCV(pipeline, params_to_tune, n_iter=num_iter, cv=5, scoring='f1_micro')
    search.fit(X, y)
    return search.best_params_


