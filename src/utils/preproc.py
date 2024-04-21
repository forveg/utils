import pandas as pd
import numpy as np
from itertools import product

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series


def process(path: str) -> pd.DataFrame:
    """Preprocessing of eneregy dataset
    Fixed procedure, no parameters
    """
    new_names = ['date', 
             'price_zone', 
             'free_transfer_zone', 
             'hour', 
             'hydro_plan',
             'nuclear_plan', 
             'thermal_plan', 
             'thermal_min', 
             'thermal_max',
             'consumption', 
             'export',
             'import',
             'solar_plan',
             'wind_plan',
             'renewable_plan',
             'supply_1',
             'supply_2',
             'price',
            ]
    
    # haven't figured out how to rename *multilevel* columns in a functional way
    def rename(df):
        df.columns = [f'{col[1]}_{col[0]}' for col in df.columns]
        return df
    
    # zero variance threshold
    variance_eps = 1e-6
    
    df = ( pd
      .read_csv(path)
      
       # rename columns for better access/readability
      .pipe(lambda df: df.rename(columns = dict(zip(df.columns, new_names))))
       
       # casting/shrinking
      .astype({'date': 'datetime64[ns]',
               'free_transfer_zone': 'category',
               'hour': 'int8'})
      .pipe(lambda df: df.astype({ col: 'float32' 
                                  for col in df.select_dtypes('float64').columns }))
      
      # drop zero-variance features
      .drop(columns = ['price_zone',
                       'renewable_plan'])
      
      # removing trailing spaces
      .pipe(lambda df: df.assign(free_transfer_zone =
              df.free_transfer_zone.cat.rename_categories(lambda s: s.strip())))
      
      # combining hours and dates into a single feature
      .pipe(lambda df: df.assign( dtime = df['date'] + pd.to_timedelta(df.hour, unit='hour')))
      
      .drop(columns = ['date', 'hour'])
            
      .pivot(index='dtime', columns='free_transfer_zone')
      .sort_index()
            
      .pipe(lambda df: df.drop(columns = df.columns[df.std(axis=0)<variance_eps]))
        
      .pipe(rename)
      #.pipe(lambda df: df.columns = [f'{col[1]}_{col[0]}' for col in df.columns])  
    )
    return df


def add_features(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Adds to the dataframe basic aggregate features, specified in kwargs
    
    Parameters
    ----------
    df : pd.DataFrame
        Source data
    kw : dict
        Dict of the following format (all keys mandatory, except for 'ewm_alphas'):
        feats = {
            'windows': [6, 12, 15, 120],
            'agg_func': ['max', 'median'],
            'ewm_alphas': [0.1, 0.5],
            'columns': ['price', 'sales_1', 'sales_2'],
        }
    
    """
    def agg_format(fun): 
        return str(fun).split(' ')[1]
    cols = kw['columns']
    
    agg_map = { 'min': np.min,
                'max': np.max,
                'median': np.median,
                'mean': np.mean,
                'std': np.std,
              }
    agg_func = [ agg_map[agg_str] for agg_str in kw['agg_func']]
    
    # level aggs
    df = pd.concat([df, pd.DataFrame(
        { f'{col}/rolling_{wind}/{agg_format(agg)}': 
         ( df[col]
          .rolling(wind)
          .aggregate(agg)
         ) for col,wind,agg in product(cols,kw['windows'],agg_func) }
    )], axis=1)
    
    # aggregates minus current value
    df = pd.concat([df, pd.DataFrame(
    { f'{col}/rolling_{wind}/{agg_format(agg)}/minus_cur':
     df[f'{col}/rolling_{wind}/{agg_format(agg)}'] - df[col]
     
     for col,wind,agg in product(cols, kw['windows'], agg_func)
    }
    )], axis=1)
    
    df = df.drop(columns = [ f'{col}/rolling_{wind}/{agg_format(agg)}' 
          for col,wind,agg in product(cols,kw['windows'],agg_func)])
    
    # ewm minus current value
    df = pd.concat([df, pd.DataFrame(
    { f'{col}/ewm_{alpha}/minus_cur':
              df[col].ewm(alpha=alpha).mean() - df[col]
              
            for col,alpha in product(cols,kw['ewm_alphas'])
    }
    )], axis=1)
    
    # calendar
    df = ( df
     .assign(month   = df.index.month,
               day   = df.index.day,
               hour  = df.index.hour,
               dayofweek = df.index.dayofweek)
     )
    
    return df

def tsfresh_extract(df: pd.DataFrame, window: int, column: str) -> pd.DataFrame:
    """Extracts tsfresh features, based on the column `column` and window size `window`
    """
    df_tmp = ( df
              .loc[:, [column]]
              .assign(dummy_id = 0)
              .reset_index()
             )
    rolled = roll_time_series(df_tmp, column_id="dummy_id", 
                              column_sort="dtime",
                              rolling_direction=1, 
                              max_timeshift=window-1, 
                              min_timeshift=window-1,
                              show_warnings=True,
                              n_jobs=4)
    feats = extract_features(rolled.drop(columns=['dummy_id']),
                             column_id="id", 
                             column_sort="dtime")
    return feats


def tsfresh_postproc(feats: pd.DataFrame, window: int, **kw) -> pd.DataFrame:
    """Postprocessing of features, extracted by tsfresh
    1. Pruning zero-variance and nan features
    2. Column/index renaming
    
    Parameters
    ----------
    feats : pd.DataFrame
        extracted features in tsfresh format
    window : int
        Window size, that was used for aggregation
    kwargs : dict
        kwargs['nan_threshold'] : float in (0; 1) (default 0.5)
            fraction of nan values in a column, that justifies its removal
        kwargs['zero_variance_threshold'] : float (default 1e-6)
            values below threshold are considered zero 
    
    Returns
    -------
    feats : pd.DataFrame
    """
    
    nan_threshold = kw.get('nan_threshold', 0.5)
    eps = kw.get('zero_variance_threshold', 1e-6)
    unrecognized = set.difference(set(kw.keys()),
                                  set(['nan_threshold', 'zero_variance_threshold']))
    if len(unrecognized)>0:
        raise ValueError(f'Unknown kwargs: {unrecognized}')
    
    feats = ( feats
             .pipe(lambda df: df.drop(columns = df.columns[ df.isna().sum(axis=0)>nan_threshold]))
             .pipe(lambda df: df.drop(columns = df.columns[ df.std()<eps]))
             .reset_index()
             
             .rename(columns={'level_1': 'dtime'})
             .set_index('dtime')
             .drop(columns='level_0')
             .rename(columns=lambda col: f'{col}_{window}')
        )
    return feats