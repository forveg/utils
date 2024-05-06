from utils.crossval import SlidingWindowCV
import numpy as np
import pytest
from itertools import product

@pytest.mark.parametrize('last_fold, ix_test',
                         [  ('keep',
                            [(200, 240), 
                             (240, 280),
                             (280, 320),
                             (320, 350)]),
                            ('drop',
                             [(200, 240), 
                              (240, 280),
                              (280, 320)]),
                            ('spread_out',
                             [(200, 250), 
                              (250, 300),
                              (300, 350)])
                           ])
def test_SlidingWindowCV_last_fold(last_fold, ix_test):

    cv = SlidingWindowCV(200, 40, 150, 20, 10, 'sequential', 1, last_fold)
    
    X = np.arange(350)[:,None]
    y = np.arange(350)

    for ( i_repeat,
            i,
            inds,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        assert np.array_equal(ix_eval_, np.arange(ix_test[i][0]-cv.gap-cv.eval_size, ix_test[i][0]-cv.gap))
        assert np.array_equal(ix_train_, np.arange(ix_test[i][0]-cv.gap-cv.train_size, ix_test[i][0]-cv.gap-cv.eval_size))

@pytest.mark.parametrize('size, ix_test',
                         [  (340,
                            [(200, 246), 
                             (246, 292),
                             (292, 340),
                             ]),
                            (322,
                             [(200, 240), 
                              (240, 280),
                              (280, 322)]),
                            (320,
                             [(200, 240), 
                              (240, 280),
                              (280, 320)]),
                            (355,
                             [(200, 251), 
                              (251, 302),
                              (302, 355)]),
                           ])
def test_SlidingWindowCV_spread_out(size, ix_test):
    cv = SlidingWindowCV(200, 40, 150, 20, 10, 'sequential', 1, 'spread_out')
    
    X = np.arange(size)[:,None]
    y = np.arange(size)

    for ( i_repeat,
            i,
            inds,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        assert np.array_equal(ix_eval_, np.arange(ix_test[i][0]-cv.gap-cv.eval_size, ix_test[i][0]-cv.gap))
        assert np.array_equal(ix_train_, np.arange(ix_test[i][0]-cv.gap-cv.train_size, ix_test[i][0]-cv.gap-cv.eval_size))

@pytest.mark.parametrize('eval_size',
                         [ 0.2,
                           0.5,
                           0.8,
                           ])
def test_SlidingWindowCV_eval_fraction(eval_size):
    ix_test = [(200, 250), 
               (250, 300),
               (300, 351)]

    cv = SlidingWindowCV(200, 40, 150, eval_size, 10, 'sequential', 1, 'spread_out')
    
    X = np.arange(351)[:,None]
    y = np.arange(351)

    for ( i_repeat,
            i,
            inds,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        assert np.array_equal(ix_eval_, np.arange(ix_test[i][0]-cv.gap-cv.eval_size, ix_test[i][0]-cv.gap))
        assert np.array_equal(ix_train_, np.arange(ix_test[i][0]-cv.gap-cv.train_size, ix_test[i][0]-cv.gap-cv.eval_size))
        assert cv.eval_size == int(np.ceil(eval_size*cv.train_size)) 

@pytest.mark.parametrize('size, ix_test',
                         [  (340,
                            [(200, 246), 
                             (246, 292),
                             (292, 340),
                             ]),
                            (322,
                             [(200, 240), 
                              (240, 280),
                              (280, 322)]),
                            (320,
                             [(200, 240), 
                              (240, 280),
                              (280, 320)]),
                            (355,
                             [(200, 251), 
                              (251, 302),
                              (302, 355)]),
                           ])
def test_SlidingWindowCV_train_minus_one(size, ix_test):
    
    cv = SlidingWindowCV(200, 40, -1, 20, 10, 'sequential', 1, 'spread_out')
    
    X = np.arange(size)[:,None]
    y = np.arange(size)

    for ( i_repeat,
            i,
            inds,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        assert np.array_equal(ix_eval_, np.arange(ix_test[i][0]-cv.gap-cv.eval_size, ix_test[i][0]-cv.gap))
        assert np.array_equal(ix_train_, np.arange(0, ix_test[i][0]-cv.gap-cv.eval_size))

@pytest.mark.parametrize('seed',
                         [1970, 99, 1861, 42])
def test_SlidingWindowCV_shuffle(seed):
    ix_test = [(200, 240), 
               (240, 280),
               (280, 320)]
    eval_size = 0.15
    cv = SlidingWindowCV(200, 40, -1, eval_size, 10, 'uniform', 1, 'spread_out', seed)
    rng = np.random.default_rng(seed)
    inds = np.arange(320)

    X = np.arange(320)[:,None]
    y = np.arange(320)

    for ( i_repeat,
            i,
            inds_,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        eval_size_true = int(np.ceil(eval_size*(ix_test[i][0]-cv.gap)))
        
        rng.shuffle(inds[:ix_test[i][0]-cv.gap])
        train_true = inds[: ix_test[i][0]-cv.gap-eval_size_true]
        eval_true  = inds[ ix_test[i][0]-cv.gap-eval_size_true : ix_test[i][0]-cv.gap]
        
        assert np.array_equal(ix_train_, train_true)
        assert np.array_equal(ix_eval_, eval_true)
        
@pytest.mark.parametrize('eval_mode, gap, eval_size', 
                            product(['sequential', 'uniform'], 
                                    [80, 21, 59], 
                                    [20, 0.2]) 
                         )
def test_SlidingWindowCV_gap(eval_mode, gap, eval_size):
    ix_test = [(200, 240), 
               (240, 280),
               (280, 320)]
    cv = SlidingWindowCV(200, 40, 120, eval_size, gap, eval_mode, 1, 'spread_out')

    X = np.arange(320)[:,None]
    y = np.arange(320)

    for ( i_repeat,
            i,
            inds,
            ix_test_,
            ix_train_,
            ix_eval_) in cv.split(X, y):
        
        assert np.array_equal(ix_test_, np.arange(ix_test[i][0], ix_test[i][1]))
        assert np.max(ix_train_)<ix_test_[0]-gap
        assert np.max(ix_eval_)<ix_test_[0]-gap