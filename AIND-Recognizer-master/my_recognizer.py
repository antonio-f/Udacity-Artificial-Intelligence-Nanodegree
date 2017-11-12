import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    # return probabilities, guesses

    # the following algorithm extracts (X, lengths) from test_set and then
    # for each pair (word,model) evaluates a score that is retained into 
    # scores dict; The list "probabilities" is updated and at the same time
    # the word with maximum score is added to "guesses" list 

    for each in test_set.get_all_Xlengths().values():
        X, lengths = each
        scores = {}
        for word, model in models.items():
            try:
                scores[word] = model.score(X, lengths)
            except:
                scores[word] = -float('inf')
        probabilities.append(scores)
        guesses.append(max(scores, key=scores.get))

    return probabilities, guesses