Practice project to display HMM maximum likelihood, encoding and decoding problem. Trained on data sets owned by a private user, so they have unfortunately not been made public. 

- MLE.py reads in an annotated training file passed through standard input command: $ python mle.py file_name.super
- Test input file in the following format: strawberry 1; orange 2; blueberry 1; ... etc
- Compatible with POS tagging, uses add-1 smoothing, "<unk>"s unknown emissions, works with an emission and transtition matrix
- Once model is created, it is saved to model.hmm

- likelihood.py uses the produced model and an observation sequence (uses space delimited tokens as observations, e.g: 1 2 1) to determine the likelihood of the observations, read through stdin: $ python likelihood.py model.hmm test.seqs
- Uses the forward and backward algorithm to determine the likelihood of the observed sequence
- Once likelihoods are calculated, output is sent to hmm.likelihood

- decoding.py uses the hmm and an observation likelihood sequence, it finds the best hidden-state sequence
- Implements the Viterbi algorithm, best sequence and its log likelihood is saved to hmm.decoding
