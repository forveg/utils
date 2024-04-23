from utils.crossval import SlidingWindowCV
import numpy as np
import pytest

"""
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
"""

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
            inds,
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
        

