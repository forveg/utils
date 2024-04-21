from utils.train import get_Xy, get_baseline
import pytest
import pandas as pd
import numpy as np

rng = np.random.default_rng(999)

@pytest.mark.parametrize( "ix, future_covar, dtime_columns",
                          [
                              (np.arange(1000), False, []),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], False, []),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], False, ['month']),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], False, ['month', 'hour']),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], True, []),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], True, ['month']),
                              (np.r_[np.arange(24, 100), np.arange(600,800)], True, ['month', 'hour']),                       
                          ])
def test_get_Xy(df,
                horiz,
                target,
                ix: int, 
                future_covar: bool,
                dtime_columns: list[str]):
    
    df = df.assign(**{col: getattr(df.index, col) for col in dtime_columns})

    X,y = get_Xy(df, ix, horiz, target, future_covar)
    
    if ix is None:
        ix = np.arange(df.shape[0])

    assert isinstance(X, pd.DataFrame) and isinstance(y, pd.Series) 
    
    # we deal with strictly monotonic slices only
    assert all(np.sort(ix)==ix)
    assert np.unique(ix).shape[0] == ix.shape[0]

    # checking NaN and not-NaN entries separately, since NaN!=NaN
    assert all(( df[target]
                .shift(-horiz)
                .iloc[ix]
                .pipe(lambda df: df.loc[df.notna()])) == y.loc[y.notna()])

    assert all(( df[target]
                .shift(-horiz)
                .iloc[ix]
                .pipe(lambda df: df.loc[df.isna()].index)) == y.loc[y.isna()].index)

    assert X.shape[0]==ix.shape[0]
    # the next check makes sense since the target was defined as an index, for convenience of debugging
    # (it's easy to see misalignments on the index column) 
    assert all(X[target]==ix) 
    assert all(X.index==df.index[ix]) # TimeIndex doesn't move, only non-index values do

    # there will be NaN values if index reaches far enough
    overreach = ix[ix >= df.shape[0]-horiz]
        
    if overreach.shape[0]>0:
        assert y.iloc[overreach].isna().all()
        if future_covar:
            assert ( X.iloc[overreach]
                    .loc[:,['pred_day', 'pred_month', 'pred_dayofweek', 'pred_hour']]
                    .isna().all().all())

    if not future_covar:
        assert all(X.columns == df.columns)
    else:
        assert all(X.columns == list(df.columns) + ['pred_month', 'pred_day', 'pred_hour', 'pred_dayofweek'])
        
        for col in ['hour', 'day', 'month', 'dayofweek']:
            # beware, TimeIndex.shift(5) shifts 5 steps FORWARD, and non-index columns shift 5 steps BACKWARD
            assert all(getattr(df.index.shift(horiz), col)[ix] == X[f'pred_{col}'])

def test_get_Xy_fails(df):
    # non-TimeIndex
    with pytest.raises(ValueError):
        get_Xy(df.reset_index(), np.arange(100), 20, 'y', True)

    # out of boundaries
    with pytest.raises(IndexError):
        get_Xy(df, np.arange(df.shape[0] + 10), 20, 'y', True)

@pytest.mark.parametrize( "specs, fillna, offset",
                    [
                        ('none', 'none', 20),
                        ('last_val', 'none', 20),
                        ('diff_median_24', 'last_val', 0),
                        ('diff_median_24', 'last_val', 20),
                        ('diff_median_24', 'last_val', 24),
                        ('diff_ewm_0.6', 'last_val', 0),
                    ])
def test_get_baseline(df, 
                      horiz,
                      target,
                      specs,
                      fillna, 
                      offset):
    ix = np.arange(offset, df.shape[0]-horiz)
    base = get_baseline(df, ix, horiz, target, specs, fillna)

    assert isinstance(base, np.ndarray)
    if specs=='none':
        assert all(base==np.zeros_like(ix))
    elif specs=='last_val':
        assert all(base==df[target].iloc[ix])
    elif specs.startswith('diff'):
        
        s = specs.split('_')
        func = s[1]
        period = float(s[2])
        if func=='median':
            agg = ( df[target]
                    .rolling(int(period))
                    .median()
                    )
        elif func=='ewm':
            agg = ( df[target]
                   .ewm(period)
                   .mean()
                   )
        elif func=='mean':
            agg = ( df[target]
                   .rolling(int(period))
                   .mean()
                   )
        agg = agg.iloc[ix]
        assert all(agg.loc[agg.notna()] == base[agg.notna().values])

        if fillna=='last_val':
            assert all(base[agg.isna().values]== ( df[target]
                                                  .iloc[ix]
                                                  .loc[agg.isna()]) )
            
        elif fillna=='none':
            assert all(np.isnan(base[agg.isna().values]))

def test_get_baseline_fails(df, 
                            horiz,
                            target):
    
    ix = np.arange(24, df.shape[0]-horiz)

    # unsupported aggregation
    with pytest.raises(ValueError):
        get_baseline(df, ix, horiz, target, 'diff_quantile_0.8',)
    
    # a typo/ unsupported specs
    with pytest.raises(ValueError):
        get_baseline(df, ix, horiz, target, 'lasst_val',)
   
    # wrong specs format
    with pytest.raises(ValueError):
        get_baseline(df, ix, horiz, target, 'median_24',)
    
