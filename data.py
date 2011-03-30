import numpy as np

class Data:
    def __init__(self, path = '.', alphabetic_baseforms = False):
        path = path + '/jhucsp.'

        # A list of all available labels, used for keying the fenome transition output densities
        self.labels = [ lbl[0:5] 
                for lbl in open(path + 'lblnames').readlines()[1:]
                if lbl != '\n' ]
        self.label_id = dict((lbl, idx) for idx, lbl in enumerate(self.labels))

        # A collection of (<word>, <instance>, <endpts>) triples, the training data
        self.instances = [ (word[:-5], 
                            map(lambda x: self.label_id[x], instance.split()),
                            map(int, endpts.split()))
                for word, instance, endpts in zip(open(path + 'trnscr').readlines()[1:],
                                                  open(path + 'trnlbls').readlines()[1:],
                                                  open(path + 'endpts').readlines()[1:]) ]

        # Prepare the recognized vocabulary
        self.vocab = list(set(map(lambda x: x[0], self.instances)))

        # Create an observation dictionary for later
        self.observations = dict()
        for word, obs, endpts in self.instances:
            if self.observations.has_key(word):
                self.observations[word].append(obs)
            else:
                self.observations[word] = [obs]
        self.observations = map(lambda word: np.asarray(self.observations[word]), self.vocab)

        # Using the list of instances, build some baseforms. These use a new set of feneme indices.
        if not alphabetic_baseforms:
            self.baseforms, self.nfenomes = default_baseforms(self.vocab, self.instances)
            self.nfenomes = len(self.labels)
        else:
            self.baseforms, self.nfenomes = alphabet_baseforms(self.vocab)

def default_baseforms(vocab, instances):
    """ Default baseforms are generated from the first observation of each
    word's acoustic representation. """

    baseforms = dict()
    maxlbl = -1
    for word, obs, endpts in instances:
        if not baseforms.has_key(word):
            # Then this is the first example, build a baseform
            bf = []
            lastlbl = -1
            for lbl in obs[(endpts[0]+1):endpts[1]]:
                maxlbl = max(maxlbl, lbl)
                if lbl != lastlbl:
                    bf.append(lbl)
                    lastlbl = lbl
            baseforms[word] = bf
    # Sort by the vocab list so we can just use indices
    return map(lambda word: baseforms[word], vocab), maxlbl+1

def alphabet_baseforms(vocab):
    """ Alphabet baseforms are generated from the spelling of the vocabulary
    term. """

    alphabet = 'abcdefghijklmnopqrstuwvxyz0123456789'
    alpha = dict( (letter, idx) 
                  for idx, letter 
                  in enumerate(alphabet) )
    out = []
    for word in vocab:
        bf = []
        lastfeneme = '-1'
        word = word.lower()
        for letter in word:
            idx = alpha[letter]
            if idx != lastfeneme:
                bf.append(idx)
                lastfeneme = idx
        out.append(bf)
    return out, len(alphabet)
