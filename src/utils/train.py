from typing import (Callable,
                    Iterator,
                    Optional )
from .crossval import (SlidingWindowCV, 
                      ParamGridIter,
                      NestedKFold )

from pprint import pprint
from datetime import datetime
from itertools import product

import shutil
import json
import pytz
import os
import argparse
import pandas as pd

from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             median_absolute_error,
                             mean_squared_error )

from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
import lightgbm as lgb

def get_baseline(df: pd.DataFrame, 
                 ix: np.ndarray,
                 horiz: int,
                 target: str,
                 specs: str,
                 fillna: str = 'last_val',
                ) -> np.ndarray:
    """Simple baseline predictions (last_value or moving average of some sort)
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series
    horizon : int (not used rn, but could be used to pick a baseline)
        Forecast horizon
    target : str 
        Target column
    specs: str
        Baseline specification string. Options:
        0 "none"     - fixed zero baseline
        1 "last_one" - last value of a time series
        2 "diff_{median|mean|ewm}_{param}" - baseline is an aggregate -
        median, mean or ewm with period=param or alpha=param in case of ewm 

    Returns
    -------
    base : np.ndarray
    """
    X = df
    if specs=='last_val':
        return X[target].iloc[ix].values
    elif specs=='none':
        return np.zeros_like(ix)

    s = specs.split('_')
    if len(s)!=3:
        raise ValueError(f'Unknown specs format: {specs}. See available options.')
    func = s[1]
    param = float(s[2])
        
    if s[0]=='diff': # Note: get rid of diff, subtraction is done outside this code
        
        if func=='median':
            param = int(param)
            base = ( X[target]
                   .rolling(param)
                   .median()
                   )
        elif func=='ewm':
            base = ( X[target]
                   .ewm(param)
                   .mean()
                   )
        elif func=='mean':
            param = int(param)
            base = ( X[target]
                   .rolling(param)
                   .mean()
                   )
        else:
            raise ValueError(f'Unsupported aggregation method: `{func}`')

        base = base.iloc[ix]

    if fillna=='last_val':
        base.loc[base.isna()] = X[target].iloc[ix].loc[base.isna()]
    else:
        assert fillna=='no_fill', f'`fillna` supports ["last_val", "no_fill"], got {fillna}'
    return base.values
                        

def get_Xy(df: pd.DataFrame,
           ix: Optional[np.ndarray],
           horiz: int,
           target: str,
           future_covar: bool = True,
           ) -> tuple[pd.DataFrame, np.ndarray] | tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Constructs X, y pair to be used in time series prediction
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series dataframe
    ix : np.ndarray or None 
        Subset of rows to retrieve
    horizon : int
        Prediction horizon (single-step prediction ``horizon`` steps ahead)
    target : str
        Target column
    future_covar : bool, default=True
        Whether to add future (calendar) covariates to X
    
    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.DataFrame]
        Training pair: y is shifted `horizon` steps back
                       X is augmented with future covariates
    """
    y = df[target].shift(-horiz)
    X = df
    
    if future_covar:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f'`future_covar`=True assumes DataFrame has DatetimeIndex. Got: {type(df.index)}')
        
        future_covars = ['month', 'day', 'hour', 'dayofweek']
        missing = [col for col in future_covars if col not in df.columns]
             
        X_future = ( df
                 .assign(**{col: getattr(df.index, col) for col in missing})   
                .loc[:, future_covars]
                .shift(-horiz)
                
                #.assign(hour_lag = np.ones(indices.shape[0])*horizon)
                .rename(columns=lambda col: 'pred_'+col)
                #.astype({'pred_'+col: int for col in future_covars})
              )
        X = pd.concat([X, X_future], axis=1)   
    if ix is not None:
        X, y = X.iloc[ix], y.iloc[ix]
    
    return X, y  

def make_callback(X_eval, y_eval,
                  X_test, y_test,
                  eval_metrics: list,
                  test_metrics: list,
                  metrics: dict[str, Callable],
                  log_freq=10):
    
    def eval_callback(env):
        if env.iteration%log_freq==0:
            eval_metrics.append(
                evaluate(y_eval, env.model.predict(X_eval), metrics))
            test_metrics.append(
                evaluate(y_test, env.model.predict(X_test), metrics))
    return eval_callback

def evaluate(y_true: np.ndarray,
             y_hat: np.ndarray,
             metrics: dict[str, Callable]) -> dict[str, float]:
    return {name: metric(y_true, y_hat) for name, metric in metrics.items()}

def fit(df: pd.DataFrame,
        cv: SlidingWindowCV,
        param_iter: Iterator[dict[str, int|float|str]],
        target: str,
        cat_feats: list[str],
        metrics: dict[str, Callable],
        horiz: int,
        offset: int,
        eval_freq: int = -1,
       **kwargs) -> tuple[list, list[lgb.LGBMRegressor]]:
    
    inds = np.arange(offset, df.shape[0]-horiz)
    X, y = get_Xy(df, inds, horiz, target, kwargs.get('future_covar', True))
    baselines = { base_model: get_baseline(df, inds, horiz, target, base_model)
                 for base_model in param_iter.param_grid['base_model'] }
    
    # lightgbm produces the same result (eg same features during bagging) 
    # unless different seeds are set manually
    seeds = np.random.choice(np.arange(int(10e6)),size=cv.n_reps)
    eval_metrics, test_metrics = [], []
    cv_res_list, models = [], []
    
    for ( (i_repeat,
           i_fold,
           _,
           ix_test_,
           ix_train_,
           ix_eval_ ),
          (i_param, 
           param_dict)) in product(cv.split(X,y), enumerate(param_iter)):

        print(f'i_rep: {i_repeat:<2}; i_param: {i_param:<2}; i_outer: {i_fold:<2}')
        
        y_resid = y.values - baselines[param_dict['base_model']]
        X_train, y_train = X.iloc[ix_train_], y_resid[ix_train_]
        X_eval,  y_eval  = X.iloc[ix_eval_],  y_resid[ix_eval_]
        X_test,  y_test  = X.iloc[ix_test_],  y_resid[ix_test_]
        
        fit_params = {
                'X': X_train,
                'y': y_train,
                'categorical_feature': cat_feats,
                'eval_set': [(X_eval, y_eval)],
                'eval_metric': ['l1'],
                'callbacks': [lgb.early_stopping(param_dict['n_early_stop_rounds'],
                                                 first_metric_only=True,
                                                 verbose=False,
                                                 min_delta=param_dict['min_delta']),]}
        eval_metrics.append([])
        test_metrics.append([])
        if eval_freq>0:
            fit_params['callbacks'].append(
                make_callback(X_eval, y_eval,
                              X_test, y_test,
                              eval_metrics[-1], test_metrics[-1],
                              {'mae': mean_absolute_error}, eval_freq))
        
        seed_params = {
            'feature_fraction_seed': seeds[i_repeat],
            'bagging_seed': seeds[i_repeat],
        }       
        model = ( lgb.LGBMRegressor(**param_dict, **seed_params)
                 .fit(**fit_params))
        
        models.append(model)
        test_loss = evaluate(y_test, model.predict(X_test), metrics)
        cv_res_list.append([i_repeat, i_fold, i_param, param_dict] +
                                    [v for v in test_loss.values()])
    if eval_freq>0:
        return cv_res_list, models, eval_metrics, test_metrics
    return cv_res_list, models

def cv_kfold(cv: RepeatedStratifiedKFold,
             n_outer_used: int,
             pg: ParamGridIter,
             X: pd.DataFrame,
             y: np.ndarray,
             metrics: dict,
             metrics_proba: dict,
             cat_cols: list[str],
             seed: int = None,
            ):
    cv_res_list, models = [], []
    seeds = np.random.choice(np.arange(int(10e6)),size=cv.n_repeats)
    # for reproducibility; this seed will be used only in the first rep
    if seed is not None:
        seeds[0] = seed
    
    n_splits = cv.get_n_splits() // cv.n_repeats
    
    for ((i, (ix_train, ix_test)),
        (i_param, param_dict)) in product(enumerate(cv.split(X,y)),
                                          enumerate(pg)):
        i_outer = i%n_splits
        i_repeat = i//n_splits

        if i_outer>=n_outer_used:
            continue

        seed_params = {
            'feature_fraction_seed': seeds[i_repeat],
            'bagging_seed': seeds[i_repeat],
        }

        #print('='*100)
        print(f'i_param: {i_param:<2}; i_outer: {i_outer:<2}; i_repeat: {i_repeat}')
        #pprint(param_dict)

        X_train, y_train = X.iloc[ix_train], y[ix_train]
        X_test,  y_test  = X.iloc[ix_test],  y[ix_test]

        model = ( lgb.LGBMClassifier(**param_dict, **seed_params)
                 .fit(X_train, y_train, categorical_feature=cat_cols))

        test_loss = evaluate(y_test, model.predict(X_test), metrics)
        test_loss_proba = evaluate(y_test, model.predict_proba(X_test)[:,1], metrics_proba)
        test_loss.update(test_loss_proba)

        cv_res_list.append([i_repeat, i_outer, i_param, param_dict] + [v for v in test_loss.values()])
        models.append(model)
        
    
    cv_res_df = pd.DataFrame(
        columns=['i_repeat', 'fold_outer', 'i_param', 'params'] + \
    ['test_'+name for name in metrics.keys()] + \
    ['test_'+name for name in metrics_proba.keys()],
    data=cv_res_list)
    
    return models, cv_res_df


def main(args):
    for path in [args.param_grid_path,
                 args.data_path,
                ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f'"{path}" doesn\'t exist')
    
    if args.weights_path is not None:
        if not os.path.exists(args.weights_path):
            raise FileNotFoundError(f'"{args.weights_path}" doesn\'t exist')
        weight_df = pd.read_parquet(args.weights_path)
    else:
        weight_df = None
    
    with open(args.param_grid_path, 'r') as f:
        param_grid = json.load(f)
    
    param_iter = ParamGridIter(param_grid)
    
    if weight_df is None and any(['weights' in name for name in param_iter.param_grid.keys()]):
        raise ValueError('param_grid specifies `weights`, but the weight file is not provided')
        
    df = pd.read_parquet(args.data_path)
    
    metric_dict = {
       'mae': mean_absolute_error,
       'mse': mean_squared_error,
       'mape': mean_absolute_percentage_error,
       'medae': median_absolute_error,
    }
    metrics = { name:metric for name,metric in metric_dict.items() if name in args.metrics }
    
    if args.cv_mode=='nested-kfold':
        cv = NestedKFold(args.n_reps, args.n_outer, args.n_inner, args.n_inner_used, args.n_outer_used)
    elif args.cv_mode=='time-series':
        cv = SlidingWindowCV(args.start, args.test_size, args.train_size, args.gap, args.n_reps)
        
        
    print('='*100)
    print('param_grid:')
    pprint(param_grid)
    print()
    print(f'{len(param_iter):<4} param combinations')
    print(f'{cv.get_n_splits(df):<4} splits')
    print(f'Total: {len(param_iter) * cv.get_n_splits(df)} models')
    
    cv_res_list, models = fit(df,
                              cv,
                              param_iter,
                              args.target_column,
                              metrics,
                              args.horiz,
                              args.offset,
                              weight_df,
                             )
    
    cv_res_df = pd.DataFrame(
            columns=['i_repeat', 'i_param', 'fold_outer', 'fold_inner', 'i_feat_elim', 'refitted', 'params'] +
            ['test_'+name for name in metrics.keys()] +
            ['eval_'+name for name in metrics.keys()],
            data=cv_res_list)
    
    tstamp = datetime.now(pytz.timezone('Europe/Samara')).strftime('%m-%d_%H:%M')

    outp = args.output_path
    if not os.path.exists(outp):
        os.makedirs(outp)

    cv_res_df.to_csv(f'{outp}/{tstamp}.csv')
    shutil.copyfile(args.param_grid_path, f'{outp}/params_{tstamp}.json')
        
    if args.save_models:
        dirname = f'{outp}/models_{tstamp}'
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for i in range(len(models)):
            models[i].booster_.save_model(f'{dirname}/model_{i:04}.txt')
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # 
    # Args prefixed with double hyphens tend to be optional,
    # but I opted for making them mandatory (which argparse allows for)
    # because it's easier to read and there's no need to keep in mind args order
    #
    parser.add_argument('--data_path',       type=str, required=True) 
    parser.add_argument('--param_grid_path', type=str, required=True)
    parser.add_argument('--target_column',   type=str, required=True)
    parser.add_argument('--output_path',     type=str, required=True)
    parser.add_argument('--metrics', nargs='*', choices=['mae', 'mse', 'mape', 'medae'], required=True)
    parser.add_argument('--cv_mode', choices=['time-series', 'nested-kfold'], required=True)
    
    required = parser.parse_known_args()[0]
    if required.cv_mode=='time-series':
        parser.add_argument('--test_size',  type=int, required=True)
        parser.add_argument('--train_size', type=int, required=True, 
                            help='train fold size. -1 to use all available data '
                            'up to the current split point')
        parser.add_argument('--start',      type=int, required=True, help='zero-based starting position of the first test fold')
        parser.add_argument('--gap',        type=int, required=True, help='gap between test and train')
        parser.add_argument('--horiz',      type=int, required=True, help='prediction horizon: steps ahead)')
        parser.add_argument('--offset',     type=int, required=True, help='number of rows at the start of the dataset '
                            'to skip (to avoid too many nan window features)')
    
    elif required.cv_mode=='nested_kfold':
        parser.add_argument('--n_inner',      type=int, required=True, help='number of inner folds')
        parser.add_argument('--n_inner_used', type=int, required=True, 
                            help='number of inner folds actually used (randomly picked). '
                            'To speed up validation, keeping folds size proportions')
        
        parser.add_argument('--n_outer',      type=int, required=True, help='number of outer folds')
        parser.add_argument('--n_outer_used', type=int, required=True, 
                            help='number of outer folds actually used (randomly picked). '
                            'To speed up validation, keeping folds size proportions')
    
    # Optionals
    parser.add_argument('--weights_path',     type=str, required=False, help='path to the table with sample weight data') 
    parser.add_argument('--n_reps',           type=int, default=1, required=False, 
                        help='number of repetitions. Repitions are needed to capture variation from '
                        'shuffling, bagging and other random factors') 
    
    parser.add_argument('--save_models', action='store_true')
    parser.add_argument('--refit',       action='store_true')
    
    args = parser.parse_args()   
    main(args)
