from utils.crossval import SlidingWindowCV, ParamGridIter
import pytest
import numpy as np

def test_sliding_window_basic():
    cv = SlidingWindowCV(700, 200, 700-10, 10, 1)
    sz = 1000
    dummy_X = np.arange(sz)[:,None]
    dummy_y = np.arange(sz)

    split = next(cv.split(dummy_X, dummy_y))
    assert isinstance(split, tuple)
    assert len(split)==7
    assert split[2]==-1


@pytest.mark.parametrize( "test_size, start, train_size, gap, n_reps, true_ix",
                         [
                             (200, 700, 700-10, 10, 1, 
                              [
                                  (np.arange(700,1000), np.arange(0,700-10), 0, 0),
                              ]),
                             (120, 700, 700-10, 10, 1,
                              [ 
                               (np.arange(700,850), np.arange(0,700-10), 0,0),
                               (np.arange(850,1000), np.arange(150, 700-10+150), 1,0)
                              ]),
                              (90, 700, -1, 10, 1,
                               [ 
                               (np.arange(700,800), np.arange(0,700-10), 0,0),
                               (np.arange(800,900), np.arange(0,700-10+100), 1,0),
                               (np.arange(900,1000), np.arange(0,700-10+100+100), 2,0),
                               ]),
                        ])
def test_sliding_window(test_size,
                        start,
                        train_size,
                        gap,
                        n_reps,
                        true_ix,
                        ):

    cv = SlidingWindowCV(start, test_size, train_size, gap, n_reps)
    sz = 1000
    dummy_X = np.arange(sz)[:,None]
    dummy_y = np.arange(sz)
    
    for (i_rep,    #0
         i_split,  #1
         _,        #2 
         perm,     #3
         ix_test,  #4
         ix_train, #5
         ix_eval), \
    (true_test, 
     true_train, 
     true_i_split, 
     true_i_rep) in zip( cv.split(dummy_X, dummy_y), true_ix):
    
        assert np.array_equal(ix_test, true_test)
        assert np.array_equal(ix_train, true_train)
        assert np.array_equal(ix_eval, ix_test) # eval is not used, should be set to ix_test
        assert true_i_split==i_split
        assert true_i_rep==i_rep

def test_paramgrid_basic():

    param_grid = {
        'feature_fraction/feature_fraction_bynode': [[0.1, 1],],
        'data_sample_strategy/bagging_freq/pos_bagging_fraction/neg_bagging_fraction/bagging_fraction': 
        [
            ['bagging',1, 1, 1.0, 1],
            ['bagging',2, 1, 0.8, 0.9],
            ['bagging',2, 1, 0.6, 1],
        ],
        'objective': ['binary'],
        'boosting': ['gbdt'],
        'learning_rate': [0.02, 0.01],
    }

    pg = ParamGridIter(param_grid)

    next1 = {'feature_fraction': 0.1,
            'feature_fraction_bynode': 1,
            'data_sample_strategy': 'bagging',
            'bagging_freq': 1,
            'pos_bagging_fraction': 1,
            'neg_bagging_fraction': 1.0,
            'bagging_fraction': 1,
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.02}
                
    next2 = {'feature_fraction': 0.1,
            'feature_fraction_bynode': 1,
            'data_sample_strategy': 'bagging',
            'bagging_freq': 1,
            'pos_bagging_fraction': 1,
            'neg_bagging_fraction': 1.0,
            'bagging_fraction': 1,
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.01
            }
    next3 = {'feature_fraction': 0.1,
            'feature_fraction_bynode': 1,
            'data_sample_strategy': 'bagging',
            'bagging_freq': 2,
            'pos_bagging_fraction': 1,
            'neg_bagging_fraction': 0.8,
            'bagging_fraction': 0.9,
            'objective': 'binary',
            'boosting': 'gbdt',
            'learning_rate': 0.02
            }

    it = iter(pg)
    assert next(it)==next1
    assert next(it)==next2
    assert next(it)==next3

    paramdf = pg.to_dataframe()
    plist = [x for x in pg]
    
    assert len(pg)==6 and len(plist)==6
    assert pg.get_fixed_params() == ['feature_fraction', 'feature_fraction_bynode', 'objective', 'boosting']
    assert paramdf.shape[0]==6 and paramdf.shape[1]==len(next1)
    assert pg.param_names == list(next1.keys())

def test_paramgrid_exceptions():

    param_grid = {
        'feature_fraction/feature_fraction_bynode': [[0.1, 1],],
        'data_sample_strategy/bagging_freq/pos_bagging_fraction/neg_bagging_fraction/bagging_fraction': 
        [
            ['bagging',1, 1, 1.0, 1],
            ['bagging',2, 1, 0.8, 0.9],
            ['bagging',2, 1, 0.6, 1],
        ],
        'objective': ['binary'],
        'boosting': ['gbdt'],
        'learning_rate': [0.02, 0.01],
        'data_sample_strategy': ['bagging', 'goss'],
    }
    
    with pytest.raises(ValueError) as e:
        ParamGridIter(param_grid)
        assert str(e)=='Duplicate key: `data_sample_strategy`'
    
    param_grid = {
        
        'objective': 'binary',
        'boosting': ('gbdt',),
        'learning_rate': [0.02, 0.01],
        'feature_fraction/feature_fraction_bynode': [[0.1, 1, 1],],
        'data_sample_strategy/bagging_freq/pos_bagging_fraction/neg_bagging_fraction/bagging_fraction': 
        [
            ['bagging',1, 1, 1.0, 1],
            ['bagging',2, 1, 0.8, 0.9],
            ['bagging',2, 0.6, 1],
        ],
    }

    with pytest.raises(ValueError) as e:
        ParamGridIter(param_grid)
        assert str(e)=='Expected entries to be lists, got `objective`: str'
    
    del param_grid['objective']    
    with pytest.raises(ValueError) as e:
        ParamGridIter(param_grid)
        assert str(e)=='Expected entries to be lists, got `boosting`: tuple'

    del param_grid['boosting']
    with pytest.raises(ValueError) as e:
        ParamGridIter(param_grid)
        key = 'feature_fraction/feature_fraction_bynode'
        assert str(e)==f'Expected param lists of the same length as key tuple, `{key}`: {param_grid[key]}'
    
    del param_grid['feature_fraction/feature_fraction_bynode']
    with pytest.raises(ValueError) as e:
        ParamGridIter(param_grid)
        
        key = 'data_sample_strategy/bagging_freq/pos_bagging_fraction/neg_bagging_fraction/bagging_fraction'
        assert str(e)==f'Expected param lists of the same length as key tuple, `{key}`: {param_grid[key]}'