.. _streaming_feature_extraction:

Streaming Feature Extraction and Learning
==========================================

Often when we are training on data it won't all fit into memory at
once. In these cases, it is advantaegous to train the in
minibatches. This is supported across many of the included classifiers
using the `partial_fit` method
 
Streaming Text Feature Extraction
---------------------------------

::

   >>> corpus = [
   ...     'This is the first document.',
   ...     'This is the second second document.',
   ...     'And the third one.',
   ...     'Is this the first document?',
   ... ]

::

   >>> n_batches = 1000
   >>> validation_scores = []
   >>> training_set_size = []
   
   >>> # Build the vectorizer and the classifier
   >>> h_vectorizer = HashingVectorizer(charset='latin-1')
   >>> clf = PassiveAggressiveClassifier(C=1)
   
   >>> # Extract the features for the validation once and for all
   >>> X_validation = h_vectorizer.transform(text_validation)
   >>> classes = np.array([-1, 1])
   
   >>> n_samples = 0
   >>> for i in range(n_batches):
   ...     
   ...    texts_in_batch, targets_in_batch = infinite_stream.next_batch()    
   ...    n_samples += len(texts_in_batch)
   ...
   ...    # Vectorize the text documents in the batch
   ...    X_batch = h_vectorizer.transform(texts_in_batch)
   ...     
   ...    # Incrementally train the model on the new batch
   ...    clf.partial_fit(X_batch, targets_in_batch, classes=classes)
   ...     
   ...    if n_samples % 100 == 0:
   ...        # Compute the validation score of the current state of the model
   ...        score = clf.score(X_validation, target_validation)
   ...        validation_scores.append(score)
   ...        training_set_size.append(n_samples)
   ...    
   ...    if i % 100 == 0:
   ...        print("n_samples: {0}, score: {1:.4f}".format(n_samples, score))

We can go a step further and continuously train new predictors on data
coming from a potentially neverending process. In this case, we never
stop training, but have better and better predictors for our program
to use as data streams in.

We simulate a neverending stream using `InfiniteStreamGenerator`
::

    >>> class InfiniteStreamGenerator(object):
    ... """Simulate random polarity queries on the twitter streaming API"""
    ...
    ...     def __init__(self, texts, targets, seed=0, batchsize=100):
    ...            self.texts_pos = [text for text, target in zip(texts, targets)
    ...                                   if target > 0]
    ...            self.texts_neg = [text for text, target in zip(texts, targets)
    ...                                   if target <= 0]
    ...            self.rng = Random(seed)
    ...            self.batchsize = batchsize
    ...
    ...         def next_batch(self, batchsize=None):
    ...            batchsize = self.batchsize if batchsize is None else batchsize
    ...            texts, targets = [], []
    ...            for i in range(batchsize):
    ...                # Select the polarity randomly
    ...                target = self.rng.choice((-1, 1))
    ...                targets.append(target)
    ...
    ...                 # Combine 2 random texts of the right polarity
    ...                pool = self.texts_pos if target > 0 else self.texts_neg
    ...                text = self.rng.choice(pool) + " " + self.rng.choice(pool)
    ...                texts.append(text)
    ...            return texts, targets



Streaming Learning
------------------------

:class:`PassiveAggressiveClassifier`
