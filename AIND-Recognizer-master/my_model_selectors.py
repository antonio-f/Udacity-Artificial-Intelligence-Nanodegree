import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores

        # * loop over components n;
        # * # of parameters: n(n-1) + n-1 + 2*n*l (l is the # of features);
        # * evaluate logL through model score function, then BIC;
        # * return best_model
        best_score = np.inf
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)
                BIC = -2 * logL + (n**2 + 2*n*len(self.X[0]) - 1) * np.log(n)
                if BIC < best_score:
                    best_score = BIC    # update best_score
                    best_model = model  # update model
            except:
                pass
        
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i))) - 1/(M-1)SUM(log(P(X(all but i)))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # # Implement model selection based on DIC scores
        best_score = -np.inf
        best_model = None
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n)               
                l = model.score(self.X, self.lengths)   # get log(P(X(i)))               
                s = 0
                for word in self.words:
                    Y, YL = self.hwords[word]
                    s += model.score(Y, YL)             # get SUM(log(P(X(i)))    
                DIC = l - (s - l)/(len(self.words)-1)   # get DIC
                if DIC > best_score:
                    best_score = DIC    # update best_score
                    best_model = model  # update model
            except:
                pass
        
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV
        
        # * use KFold to create train and test indices;
        # * define pairs (X, lenghts) for training and test;
        # * score using the model defined by n;
        # * evaluate the mean of the scores obtained;
        # * when loop ends, best_model is returned if possible 

        best_score = -np.inf
        best_model = None

        for n in range(self.min_n_components, self.max_n_components+1):  # loop over #components
            try: # use KFold for train and test indices; define pairs (X, lenghts) for training and test;
                scores = []
                for cv_train_idx, cv_test_idx in KFold(n_splits=7).split(self.sequences):
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    model = self.base_model(n) # score 
                    scores.append(model.score(X_test, lengths_test))
                score = np.mean(scores) # evaluate mean of scores
                if score > best_score:
                    best_score = score # update score
                    best_model = model # update model
            except:
                pass

        return best_model if best_model else self.base_model(self.n_constant)
