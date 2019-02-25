
import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # return probabilities, guesses
    all_xlengths = test_set.get_all_Xlengths()
    for _, data_tuple in all_xlengths.items():
        x_vals, lengths = data_tuple
        words = {}
        for word, model in models.items():
            try:
                words[word] = model.score(x_vals, lengths)
            # pylint: disable=broad-except
            # exceptions vary and occurs deep in other external classes
            except Exception:
                words[word] = float('-inf')
        probabilities.append(words)

    guesses = [max(probs, key=probs.get)
               for probs in probabilities]
    return probabilities, guesses
