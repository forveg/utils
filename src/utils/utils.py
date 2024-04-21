import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

from .train import get_Xy, get_baseline

def q95(x):
    return np.quantile(x, 0.95)

def summary(cv_res: pd.DataFrame, 
            aggregates: list = ['median', 'std']) -> pd.DataFrame:
    gr = cv_res.groupby('i_param')
    metrics = [ col for col in cv_res.columns if 'test' in col]
    
    return ( pd.concat([
        # (1) index-dataframe
        pd.DataFrame(index = gr.groups.keys(),
                     data  = {'i_param':gr.groups.keys()}),
        # (2) aggregates-dataframe
        gr.agg({ col: aggregates for col in metrics }),
        
        # (3) flatten-dictionary-dataframe 
        pd.DataFrame(index = gr.groups.keys(),
                     data  = [x for x in gr['params'].agg(lambda df: eval(str(df.iloc[0])))]),
    ], axis=1)
            # aggregate column names are tuples. Change them to strings
            .rename(columns = lambda c: c if isinstance(c,str) else f'{c[0]}/{c[1]}')
           )

def feat_importance(model, silent=False):
    if isinstance(model, lgb.LGBMRegressor):
        model = model.booster_
        
    trees = model.trees_to_dataframe()
    groups = trees.groupby('split_feature')
    
    imp = ( pd
           .DataFrame(index = model.feature_name())
           .assign(
              gain   = groups['split_gain'].sum(),
              splits = groups['split_gain'].count(),
              depths = groups['node_depth'].aggregate(lambda x: list(x)),
              counts = groups['count'].aggregate(lambda x: list(x)),
           )
           .fillna(0)
           .astype({'splits': int})
          )
    
    if not silent:
        # more stats?
        print(f'{(imp.splits==0).sum()}/{imp.shape[0]} unused')
    return imp

def hist_metrics(res_df: pd.DataFrame, metric: str) -> None:
    groups = res_df.groupby('i_param')
    n_groups = groups.size().shape[0]

    fig,ax=plt.subplots(n_groups, 1, figsize=(10,n_groups*0.5), sharex=True)
    for i in range(n_groups):       
        g = groups.get_group(i)
        
        #cnts, bins = np.histogram(g[metric], bins=30)
        #ax[i].bar(bins[1:], cnts/np.sum(cnts), (bins[-1]-bins[0])/50, alpha=0.5)
        ax[i].hist(g[metric], bins=30, alpha=0.5)
        ax[i].axvline(g[metric].median(), lw=0.8, c='g')
        ax[i].set_ylabel(i)
        plt.tight_layout(h_pad=0)


def barplot_metrics(res_df: pd.DataFrame, metric: str) -> None:
    groups = res_df.groupby('i_param')
    n_groups = groups.size().shape[0]
    n_folds = res_df.fold_outer.max()+1
    
    best = res_df.loc[res_df.fold_outer==res_df.fold_outer.max(), metric].min()
    #best_fold0 = res_df.loc[res_df.fold_outer==0, metric].min()

    palette = ['lightcoral', 'bisque', 'greenyellow', 'skyblue', 'lightsteelblue']
    colors = [palette[ix] for ix in range(n_folds)]
    fig,ax=plt.subplots(n_groups, 1, figsize=(7,n_groups*0.3), sharex=True)
    for i in range(n_groups):
        
        g = groups.get_group(i)
        ax[i].bar(g[metric], 0.2, 0.1, color=colors)
       # ax[i].bar(res_df.loc[(res_df.i_param==i) & (res_df.fold_outer==res_df.fold_outer.max())][metric],
       #           0.2, 0.1, color='green', alpha=0.4)
        ax[i].bar(g.loc[g.fold_outer==n_folds-1, metric],
                  0.2, 0.1, color='green')

        
        ax[i].axvline(best, color='green', lw=0.8, linestyle=(0,(5,5)))
        #ax[i].axvline(best_fold0, color='salmon', lw=0.8)

        ax[i].set_yticks([])
        ax[i].set_ylabel(i)
        #if i!=n_groups-1: ax[i].set_xticks([])
        plt.tight_layout(h_pad=0)
        
    #ax[0].legend(labels=[f'fold_{i}' for i in range(res_df.fold_outer.max())])

def display_predict(df, models, pred_start, horiz, target, margin=40, scatter=False, legend_params=None):
    fig,ax=plt.subplots(figsize=(12,3))
    dtime = df[target].iloc[pred_start-margin : pred_start+horiz].index
    ax.plot(dtime, df[target].iloc[pred_start-margin : pred_start+horiz].values)
    
    if isinstance(models[0], lgb.LGBMRegressor):
        models = [m.booster_ for m in models]
    for m in models:
        X,_ = get_Xy(df, np.arange(pred_start-horiz, pred_start), horiz, target)
        base = get_baseline(df, 
                            np.arange(pred_start-horiz, pred_start), 
                            horiz, 
                            target,
                            m.params['base_model'])
        y_hat = base + m.predict(X)
        if scatter:
            ax.scatter(dtime[margin: margin+horiz], y_hat, alpha=0.5, s=2)
        else:
            ax.plot(dtime[margin: margin+horiz], y_hat, alpha=0.5)
    labels = ['target']
    if legend_params:
        labels += [' '.join([str(m.params[p]) for p in legend_params]) for m in models]
    else:
        labels += [str(x) for x in np.arange(len(models))]
    ax.legend(labels)
    
def load_models(timestamp: str, 
                output_path: str,
                sort_by: tuple[str,str], 
                top_k: int=10, 
                ) -> tuple[ list[list[lgb.Booster]], list[dict]]:
    """Load saved models and their respective param_dicts
    
    Parameters
    ----------
    timestamp : str 
        
    sort_by : tuple
        Column to sort models by: eg ('test_mae','median'), ('test_mape', 'q95')
    top_k : int
        How many top models to load
    
    Returns
    -------
    (boosters, param_dicts)
        boosters is a list of lists of lgb.Boosters - each resulted from CV of a single 
        parameter set: different models correspond to different folds and repititions
        
        param_dicts is a list of param dictionaries for each model
    """
    res_df = pd.read_csv(f'{output_path}/{timestamp}.csv', index_col=0)
    sm = summary(res_df)
    if sort_by not in sm.columns:
        raise ValueError(f'`{sort_by}` not in model_results')
    
    # indices of best parameter sets
    best_i_params =  ( sm
                      .sort_values(sort_by)
                      .iloc[:top_k]
                      .i_param
                      .values )
    
    # indices of individual models corresponding to best params (evaluated on different folds)
    inds = [ res_df.loc[res_df.i_param==i_param].index 
            for i_param in best_i_params ]
    
    # individual models
    models = [ [ lgb.Booster(model_file=f'{output_path}/models_{timestamp}/model_{i:04}.txt')
                for i in inds_for_single_param] for inds_for_single_param in inds]

    param_dicts = [ eval(res_df.iloc[i].params) for i in [ix[0] for ix in inds] ]
    
    #if n_estimators is None:
    #    print('Since `n_estimators` is not provided, for models with the number of iterations varying across cv folds, *the first fold\'s* number of iterations would be chosen by default')
    #else:
    #    assert isinstance(n_estimators, list) and len(n_estimators)==top_k, '`n_estimators` has to be a list of size `top_k`'
    
    for i in range(len(models)):
        #print(best_i_params[i], [m.num_trees() for m in models[i]])
        #param_grids[i]['n_estimators'] = models[i][0].num_trees() if n_estimators is None else n_estimators[i] 
        
        # my custom params should be copied separately
        for m in models[i]:
            m.params['base_model'] = param_dicts[i]['base_model']
            m.params['n_early_stop_rounds'] = param_dicts[i]['n_early_stop_rounds']
    
    return models, param_dicts
