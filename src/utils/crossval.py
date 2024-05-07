import pandas as pd    
import numpy as np    

from itertools import product
from collections import defaultdict
from functools import reduce
from typing import Iterator, TypeAlias
from sklearn.model_selection import KFold, StratifiedKFold

class SlidingWindowCV:  

    def __init__(self, 
                 start: int,
                 test_size: int,
                 train_size: int,
                 eval_size: int | float,
                 gap: int,
                 eval_mode: str = 'sequential',
                 n_reps: int = 1,
                 last_fold: str = 'spread_out',
                 seed = None) -> None:    
        """Sliding window split generator a-la sklearn. More finely customizable
        See https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split for illustration 
        
        Parameters
        ----------
        start : int
            Position of the first test fold
        test_size : int
            *Nominal* test fold size (see below)
        train_size : int 
           Total train fold size, *including* eval fold.
           Set train_size=-1 to use all available past data
        eval_size : int | float 
           Eval size: if int, must be eval_size<train_size
                      if float, must be 0<eval_size<1, fraction of train_size
        gap : int
            The gap between train and test split - to exclude highly correlated (adjacent) regions
        eval_mode : str {'sequential', 'shuffle', 'tail-shuffle}
            If 'sequential', eval fold is located at the end of train fold;
            If 'shuffle', eval fold is a random subset of train fold
            If 'tail-shuffle', train and eval are random subsets of all available data
            up to this point (ie at any point train+eval size is fixed and equal to `train_size`,
            but it gets sampled from the entire tail)
        n_reps : int (default 1)
            Number of repititions.
        last_fold : str {'spread_out', 'keep', 'drop'}
            'spread_out': is the last fold is smaller than `test_size`,
            spread it evenly across other folds. If there is a remainder,
            add it to the last fold;

            'keep': keep last smaller fold as is;
            'drop': drop last smaller fold  
        
        Example
        -------
        start      = 13
        test_size  = 5
        train_size = 8
        eval_size  = 3
        gap        = 2
        eval_mode  = 'sequential'
        
        would yield the following partitions for the series of length 24:
        
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 
        -------------------------------------------------------------------------
        0          x0 x0 x0 x0 x0 e0 e0 e0 _  _  y0 y0 y0 y0 y0       
        1                         x1 x1 x1 x1 x1 e1 e1 e1 _  _  y1 y1 y1 y1 y1 y1             
       
        where x%d, e%d, y%d, _ denote train, eval, test and gap respectively.
        Test fold slides forward, starting at ``start`` position, with implicit stride=test_size.
       
        """
        if train_size+gap>start:
            raise ValueError('(`train_size` + `gap`) must be less than or equal to `start`')
        
        self.start = start
        self.test_size = test_size
        
        if train_size<=0 and train_size!=-1:
            raise ValueError('Expected `train_size`>0 or `train_size`=-1')
        if isinstance(eval_size, float) and (eval_size<=0 or eval_size>=1):
            raise ValueError('Expected float `eval_size` in (0; 1)')
        
        self.train_size = train_size
        if train_size==-1:
            self.eval_size = eval_size
        elif isinstance(eval_size, float):           
            self.eval_size = int(np.ceil(eval_size*train_size))
        else:
        # train_size>0 and eval_size is int
            if eval_size<=0:
                raise ValueError('Expected positive `eval_size`')
            if eval_size >= train_size:
                raise ValueError('`eval_size` has to be less than `train_size`')
            self.eval_size = eval_size
        
        if self.train_size!=-1 and (not isinstance(self.eval_size, float)) and \
        (start < self.train_size + gap):
            raise ValueError('`start` has to be greater than `train_size`+`gap`')
        
        eval_modes = ['shuffle', 'tail-shuffle', 'sequential']
        if eval_mode not in eval_modes:
            raise ValueError(f'Expected `eval_mode` in {eval_modes}, got: {eval_mode}')
        self.eval_mode = eval_mode
        self.gap = gap
        self.n_reps = n_reps
        self.seed = seed

        if last_fold not in ['spread_out', 'drop', 'keep']:
            raise ValueError("Expected `last_fold` in {'spread_out', 'drop'}")       
        self.last_fold = last_fold

    def _get_test_sizes(self, sz: int) -> np.ndarray:
        """Calculates actual sizes of test folds
        
        Parameters
        ----------
        sz : int
            Length of time series
        start: int
            Position of the first test fold
        test_size : int
            Nominal test fold size
        
        Returns
        -------
        test_sizes : np.ndarray
            Actual sizes of test folds
        """
        n_splits = int((sz - self.start)//self.test_size)
        test_sizes = np.ones(n_splits, dtype=np.int16)*self.test_size
        rem = (sz-self.start) % self.test_size
            
        if self.last_fold=='spread_out':
            test_sizes += rem//n_splits
            test_sizes[-1] += rem%n_splits
        elif self.last_fold=='keep':
            test_sizes = np.r_[test_sizes, rem]
        return test_sizes
    
    def validate_input(self, X: pd.DataFrame) -> None:
        sz = X.shape[0]
        if self.start>sz or self.start + self.test_size>sz:
            raise ValueError('Input is too small for the sizes provided: '
                            f'start: {self.start}, test: {self.test_size}, X: {sz}')
        
    
    SplitIter : TypeAlias = Iterator[tuple[int, int, int, 
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    
    def split(self, X: pd.DataFrame | np.ndarray,
              y: pd.DataFrame | np.ndarray) -> SplitIter:
        """Iterates over splits
        
        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            X of the training pair
        y : pd.DataFrame | np.ndarray
            target of training pair, not used
        
        Returns
        -------
        split : tuple of the form
            ( repetition_index,
              split_index,
              -1, # unused
              permutation, # unused
              test_index,
              train_index,
              eval_index # equals test_index, unused
            )
            Unused fields are for compatibility
        """
        self.validate_input(X)
        sz = X.shape[0]
        test_sizes = self._get_test_sizes(sz)
        
        if any(test_sizes!=self.test_size):
            print('Samples from the last (incomplete) fold were spread out: '
                 f'resulting folds: {list(test_sizes)}')

        rng = np.random.default_rng(self.seed)

        n_splits = test_sizes.shape[0]
        inds = np.arange(sz)           
        for i_repeat in range(self.n_reps):
            ix = self.start
            for i in range(n_splits):

                ix_test_  = inds[ix : ix+test_sizes[i]]
                train_size = self.train_size
                if self.train_size==-1:
                    train_size = ix - self.gap
                    
                eval_size = self.eval_size
                if isinstance(self.eval_size, float):
                    eval_size = int(np.ceil(self.eval_size * train_size))  
                
                train_size = train_size - eval_size

                if self.eval_mode=='tail-shuffle':
                    rng.shuffle(inds[ix - self.gap - eval_size - train_size: ix - self.gap])
                elif self.eval_mode=='shuffle':
                    inds = np.arange(sz)
                    rng.shuffle(inds[ix - self.gap - eval_size - train_size: ix - self.gap])

                ix_train_ = inds[ix - self.gap - train_size - eval_size: ix - self.gap - eval_size]    
                ix_eval_ = inds[ix - self.gap - eval_size: ix - self.gap]
                        
                yield ( i_repeat,
                           i,
                           inds,
                           ix_test_,
                           ix_train_,
                           ix_eval_, 
                          )
                ix += test_sizes[i]

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """Returns total number of splits
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series dataframe. Number of splits is unknown in advance,
            it depends on the position of ``start`` within series
        
        Returns
        -------
        n_splits : int
            Number of splits
        """
        self.validate_input(X)
        return int((X.shape[0] - self.start)//self.test_size) * self.n_reps    

class SlidingWindowCV_legacy:  

    def __init__(self, 
                 start: int,
                 test_size: int,
                 train_size: int,
                 gap: int,
                 n_reps: int = 1) -> None:    
        """Sliding window split generator a-la sklearn. More finely customizable
        See https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split for illustration 
        
        Parameters
        ----------
        start : int
            Position of the first test fold
        test_size : int
            *Nominal* test fold size (see below)
        train_size : int (set train_size = -1 to use all available past data)
            Train fold size
        gap : int
            The gap between train and test split - to exclude highly correlated (adjacent) regions
        
        Example
        -------
        start      = 13
        test_size  = 5
        train_size = 8
        gap        = 2
        
        would yield the following partitions for the series of length 24:
        
          00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 
        -------------------------------------------------------------------------
        0          x0 x0 x0 x0 x0 x0 x0 x0 _  _  y0 y0 y0 y0 y0       
        1                         x1 x1 x1 x1 x1 x1 x1 x1 _  _  y1 y1 y1 y1 y1 y1             
       
        where x%d, y%d, _ denote train, test and gaps respectively.
        Test fold slides forward, starting at ``start`` position, with implicit stride=test_size.
       
        Note, if a time series fits non-integer number of test folds (eg series of length 13 and 
        test_size = 5), then the last smaller fold would be spread over bigger ones 
        (eg 5,5,3 would result in 6,7).
        """
        if train_size+gap>start:
            raise ValueError('(`train_size` + `gap`) must be less than or equal to `start`')
        
        self.start = start
        self.test_size = test_size
        self.train_size = train_size
        self.gap = gap
        self.n_reps = n_reps
        
    def _get_test_sizes(self, sz: int) -> np.ndarray:
        """Calculates actual sizes of test folds
        
        Parameters
        ----------
        sz : int
            Length of time series
        start: int
            Position of the first test fold
        test_size : int
            Nominal test fold size
        
        Returns
        -------
        test_sizes : np.ndarray
            Actual sizes of test folds
        """
        n_splits = int((sz - self.start)//self.test_size)
        test_sizes = np.ones(n_splits, dtype=np.int16)*self.test_size
        rem = (sz-self.start) % self.test_size
        test_sizes += rem//n_splits
        test_sizes[-1] += rem%n_splits
        return test_sizes
    
    def validate_input(self, X: pd.DataFrame) -> None:
        sz = X.shape[0]
        if self.start>sz or self.start + self.test_size>sz:
            raise ValueError('Input is too small for the sizes provided: '
                            f'start: {self.start}, test: {self.test_size}, X: {sz}')
    
    SplitIter : TypeAlias = Iterator[tuple[int, int, int, 
                                           np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    
    def split(self, X: pd.DataFrame | np.ndarray,
              y: pd.DataFrame | np.ndarray) -> SplitIter:
        """Iterates over splits
        
        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            X of the training pair
        y : pd.DataFrame | np.ndarray
            target of training pair, not used
        
        Returns
        -------
        split : tuple of the form
            ( repetition_index,
              split_index,
              -1, # unused
              permutation, # unused
              test_index,
              train_index,
              eval_index # equals test_index, unused
            )
            Unused fields are for compatibility
        """
        self.validate_input(X)
        sz = X.shape[0]
        
        test_sizes = self._get_test_sizes(sz)
        if any(test_sizes!=self.test_size):
            print('Samples from the last (incomplete) fold were spread out: '
                 f'resulting folds: {list(test_sizes)}')
            
        n_splits = test_sizes.shape[0]
        perm = np.arange(sz)           
        for i_repeat in range(self.n_reps):
            ix = self.start
            for i in range(n_splits):

                ix_test_  = perm[ix : ix+test_sizes[i]]
                if self.train_size==-1:
                    ix_train_ = perm[ : ix - self.gap]
                else:
                    ix_train_ = perm[ix - self.train_size - self.gap : ix - self.gap]
                ix_eval_ = ix_test_
                yield ( i_repeat,
                           i,
                           -1,
                           perm,
                           ix_test_,
                           ix_train_,
                           ix_eval_, 
                          )
                ix += test_sizes[i]

    def get_n_splits(self, X: pd.DataFrame) -> int:
        """Returns total number of splits
        
        Parameters
        ----------
        X : pd.DataFrame
            Time series dataframe. Number of splits is unknown in advance,
            it depends on the position of ``start`` within series
        
        Returns
        -------
        n_splits : int
            Number of splits
        """
        self.validate_input(X)
        return int((X.shape[0] - self.start)//self.test_size) * self.n_reps
    
class NestedKFold:
    def __init__(self, n_repeats, n_outer, n_inner, n_outer_used, n_inner_used, seed=None):
        
        self.n_repeats = n_repeats
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_outer_used = n_outer_used
        self.n_inner_used = n_inner_used
        self.seed = seed
    
    def split(self, X, y):
        rng = np.random.default_rng(self.seed)
        perm = np.arange(X.shape[0])
        outer_cv = KFold(n_splits=self.n_outer, shuffle=False)
        inner_cv = KFold(n_splits=self.n_inner, shuffle=False)
        
        for i_repeat in range(self.n_repeats):      

            rng.shuffle(perm)           
            for i_outer, (ix_inner, ix_test) in enumerate(outer_cv.split(X, y)):
                
                if i_outer>=self.n_outer_used:
                    break
                ix_test_ = perm[ix_test]
                for i_inner, (ix_train, ix_eval) in enumerate(inner_cv.split(X.iloc[ix_inner],
                                                                            y.iloc[ix_inner])):
                    if i_inner>=self.n_inner_used:
                        break
                    ix_train_ = perm[ix_inner[ix_train]]
                    ix_eval_  = perm[ix_inner[ix_eval]]

                    yield ( i_repeat,
                                   i_outer,
                                   i_inner,
                                   perm,
                                   ix_test_,
                                   ix_train_,
                                   ix_eval_, 
                                  )
    def get_n_splits(self):
        return self.n_repeats * self.n_inner_used * self.n_outer_used

class StratifiedNestedKFold:
    def __init__(self, n_repeats, n_outer, n_inner, n_outer_used=-1, n_inner_used=-1, shuffle=True, seed=None):
        """
        Parameters
        ----------
        n_repeats : int 
            cf sklearn.RepeatedKFold;
            Repititions are needed to capture variance from random split generation.
            Each repitition has a separate seed.
        n_outer : int
            Number of outer (test) folds 
        n_inner : int
            Number of inner (eval) folds
        n_outer_used : int (default -1)
            Number of outer folds actually used. Eg, when `n_outer`=6 and `n_outer_used`=5
            only first 5 outer folds would be used. Used to speed up (relax) cross validation,
            while preserving fold sizes (single test fold is 1/6 of data size).
            Use all by default. 
        n_inner_used : int (default -1)
            See above
        shuffle : bool (default True)
            Whether to shuffle data or go through all consecutive chunks in order
        seed : None | int (default None)
            Seed for `np.random.default_rng`        
        """
        self.n_repeats = n_repeats
        self.n_outer = n_outer
        self.n_inner = n_inner
        
        self.n_outer_used = n_outer if n_outer_used==-1 else n_outer_used
        self.n_inner_used = n_inner if n_inner_used==-1 else n_inner_used
        
        self.seed = seed
        self.shuffle = shuffle
    
    def _check_input(self, x, name):
        if isinstance(x, pd.DataFrame):
            res = x.values
        elif isinstance(x, np.ndarray):
            res = x
        else:
            raise ValueError(f'unsupported type, {name}: {type(x)}')
        return res
    
    def split(self, X : pd.DataFrame | np.ndarray,
               y : pd.DataFrame | np.ndarray):
        '''
        Returns
        -------
        Yields the following tuple on each iteration
        ( i_repeat,
          i_outer,
          i_inner,
          perm,
          ix_test,
          ix_train,
          ix_eval )
        '''
        
        X_ = self._check_input(X, 'X')
        y_ = self._check_input(y, 'y')

        rng = np.random.default_rng(self.seed)
        perm = np.arange(X_.shape[0])
        outer_cv = StratifiedKFold(n_splits=self.n_outer, shuffle=False)
        inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=False)
        
        for i_repeat in range(self.n_repeats):      
            
            if self.shuffle:
                rng.shuffle(perm)           
            for i_outer, (ix_inner, ix_test) in enumerate(outer_cv.split(X_, y_)):
                
                if i_outer>=self.n_outer_used:
                    break
                ix_test_ = perm[ix_test]
                for i_inner, (ix_train, ix_eval) in enumerate(inner_cv.split(X_[ix_inner],
                                                                            y_[ix_inner])):
                    if i_inner>=self.n_inner_used:
                        break
                    ix_train_ = perm[ix_inner[ix_train]]
                    ix_eval_  = perm[ix_inner[ix_eval]]

                    yield ( i_repeat,
                                   i_outer,
                                   i_inner,
                                   perm,
                                   ix_test_,
                                   ix_train_,
                                   ix_eval_, 
                                  )
    def get_n_splits(self):
        return self.n_repeats * self.n_inner_used * self.n_outer_used

class ParamGridIter:
    """Generator of hyperparameter grid combinations"""
    
    def __init__(self, param_grid: dict) -> None:
        """
        Parameters
        ----------
        param_grid : dict
            Hyperparameter grid of a particular format.
            Example:
            
            param_grid = {
                    'data_sample_strategy': ['bagging', 'goss'],
                    'objective': ['mae', ],
                    'boosting': ['gbdt'],
                    'feature_fraction': [0.3, 0.7],
                    'learning_rate': [0.02,],
                    'n_estimators': [200,],
                    
                    'max_depth/num_leaves': [[7, 80], [7, 100], [6, 60]],
                    'bagging_fraction/bagging_freq': [[0, 0], [0.7, 5]],
                    
                    'verbose': [4],
                    'n_jobs': [4],
            }
            
            Dict keys are either parameter names or parameter names, separated
            by slash `/`. Dict values are always lists, even if singletons.
            
            If the key is a single name, then all elements from its list will be combined 
            with combinations of all other parameters (producing cartesian product)
            
            If the key includes multiple names, then elements of its list are tuples of the same
            arity -- all possible combinations of those parameters.
            *The point is to specify only those combinations, that make sense, and to avoid
            looping over entire cartesian product* 
            
            E.g. full product of ``max_depth`` and ``num_leaves`` would look like
            
            [(5, 28), (5, 100), (5, 150), (7, 28), (7, 100), (7, 150) ...]
            of which many make no sense.
        """
        
        self.param_grid = param_grid 
        self.param_names = reduce(lambda x,acc: x+acc, 
                                    [x.split('/') for x in param_grid.keys()],
                                    [])  
        self.validate_input()
        self.param_cnt = reduce(lambda x,y:x*y, 
                                    [len(params) for params in param_grid.values()], 1)
    
    def _reset_param_gen(self, param_grid) -> None:
        return ( reduce(lambda x,acc: x+acc,
                              [x if isinstance(x,list) else [x] for x in param_set],
                              []) 
                             for param_set in product(*param_grid.values())) 
        
    
    def validate_input(self) -> None:
        """Non-exhaustive check of param_grid validity
        
        Throws exception in case of failure.
        """
        counts = defaultdict(int)
        for name in self.param_names:
            counts[name]+=1
            if counts[name]>1:
                raise ValueError(f'Duplicate key: `{name}`')
        for k,v in self.param_grid.items():
            if not isinstance(v, list):
                raise ValueError(f'Expected entries to be lists, got `{k}`: {type(v)}')
            if '/' in k:
                if any([not isinstance(x,list) for x in v]):
                    raise ValueError(f'Expected list of lists, `{k}`: {v}')
                if any([len(x)!=len(k.split('/')) for x in v]):
                    raise ValueError(f'Expected param lists of the same length as key tuple, `{k}`: {v}')
    
    def to_dataframe(self, drop_const=False):
        """
        Parameters
        ----------
        drop_const : bool (default: False)
            Whether to omit parameters with a single possible combination
        """
        fixed_params = self.get_fixed_params() if drop_const else []
        return ( pd.DataFrame(data    = [param for param in iter(self)],
                              columns = self.param_names)
                .pipe(lambda df: df.drop(columns=fixed_params)))
    
    def __len__(self) -> int:
        """Returns total number of parameter combinations
        """
        return self.param_cnt
    
    def __iter__(self) -> Iterator[dict]:
        for v in product(*list(self.param_grid.values())):
            yield dict(zip(self.param_names,
                           # flatten mixed lists like [0.1, [1, 'gbdt'], 10, ['bagging', 0.8, 0.7]]
                           [x for sublist in v for x in (sublist if isinstance(sublist, list) else [sublist])]))

    def get_fixed_params(self):
        fixed_params = []
        for k,v in self.param_grid.items():
            if len(v)==1:
                fixed_params += k.split('/')
        return fixed_params
