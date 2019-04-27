"""
Skip-gram model.
"""
import logging
import math
from typing import Iterable, List, Tuple, Optional, Union

import numpy as np
import tensorflow as tf
import tqdm


from word2vec.exceptions import Word2VecException
from word2vec.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


class SkipGramModel:

    def __init__(
            self,
            embedding_size: int,
            context_window: Union[int, Tuple[int, int]] = 5,
            context_samples: Union[str, int] = 1,
            negative_samples: int = 5,
            stride: int = 1,
    ):
        """
        Parameters
        ----------
        embedding_size: desired word embedding size

        context_window: shape of the context window
            The words on the left and on the right side of a source word
            constitute a context. The shape can be specified as a tuple of
            integers (<left>, <right>), or a single integer <n> which is
            equivalent to (<n>, <n>).

        context_samples: number of positive samples to take from the context
            The parameter can be an a positive integer or the string "all".
            In the first case a number of random samples is taken within
            the context window. In the second case each target word in the
            context is used exactly once.

        negative_samples: number of negative samples to use

        stride: number of steps to move the context window while scanning the input

        """
        self.embeddings: np.ndarray = np.array([])
        self.vocabulary: Vocabulary = Vocabulary()

        self.embedding_size: int = embedding_size
        self.context_window: Tuple[int, int] = self._context_window_shape(context_window)
        self.context_samples: Optional[int] = self._context_samples(context_samples)
        self.negative_samples: int = negative_samples
        self.stride: int = stride

        self._context_indices = np.array(
            [i for i in range(-self.context_window[0], 0)] +
            [i for i in range(1, self.context_window[1] + 1)]
        )

    @staticmethod
    def _context_window_shape(context_window: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Get the shape of the context window.
        """
        exception = Word2VecException("Unsupported type of context window: %s" % str(context_window))

        # Single integer.
        if isinstance(context_window, int):
            if context_window < 1:
                raise exception
            return context_window, context_window

        # Pair of integers.
        if isinstance(context_window, (tuple, list)):
            if len(context_window) != 2:
                raise exception
            if not isinstance(context_window[0], int) or context_window[0] < 0:
                raise exception
            if not isinstance(context_window[1], int) or context_window[1] < 0:
                raise exception
            return context_window

        # Unsupported type.
        raise exception

    @staticmethod
    def _context_samples(context_samples: Union[str, int]) -> Optional[int]:
        """
        Number of positive samples to take from the context.
        """
        exception = Word2VecException("Unsupported type of context samples: %s" % str(context_samples))

        # Integer.
        if isinstance(context_samples, int):
            if context_samples < 1:
                raise exception
            return context_samples

        # String.
        if isinstance(context_samples, str):
            if not context_samples == "all":
                raise exception
            return None

        # Unsupported type.
        raise exception

    def _random_target_index(self, source_index: int) -> int:
        """
        Chose a random index in the context window around a source index.
        """
        return source_index + np.random.choice(self._context_indices, 1)[0]

    def _target_indices(self, source_index: int) -> List[int]:
        """
        Return all target indices in the context window around a source index.
        """
        return [source_index + i for i in self._context_indices]

    def _generate_training_set(
            self,
            transformed_texts: List[List[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        Generate a training set.

        Parameters
        ----------
        transformed_texts: texts in form of integer sequences

        Returns
        -------
        inputs, labels: inputs and labels in form of integer sequences
        """
        inputs: List[int] = []
        labels: List[int] = []

        for tokens in transformed_texts:
            start = self.context_window[0]
            stop = len(tokens) - self.context_window[1]

            # Slide the context window over the text.
            for source_index in range(start, stop, self.stride):

                # Use all words in the context window.
                if self.context_samples is None:
                    for target_index in self._target_indices(source_index):
                        inputs.append(source_index)
                        labels.append(target_index)

                # Or take random samples from the context.
                else:
                    for _ in range(self.context_samples):
                        inputs.append(source_index)
                        labels.append(self._random_target_index(source_index))

        return inputs, labels

    @staticmethod
    def _generate_batches(
            inputs: List[int],
            labels: List[int],
            batch_size: int
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate batches from a training set.

        Parameters
        ----------
        inputs: inputs in form of an integer sequence
        labels: labels in form of an integer sequence
        batch_size: desired batch size

        Returns
        -------
        inputs_batches, labels_batches: lists of numpy array
            The shapes of input batch arrays are (<batch_size>, ).
            The shapes of label batch arrays are (<batch_size>, 1).
        """
        inputs_batches: List[np.ndarray] = []
        labels_batches: List[np.ndarray] = []

        for i in range(math.ceil(len(inputs) / batch_size)):

            # Select batches.
            start = i * batch_size
            end = (i + 1) * batch_size
            inputs_batch: List[int] = inputs[start:end]
            labels_batch: List[int] = labels[start:end]

            # Pad with zeros if batches too small.
            if len(inputs_batch) < batch_size:
                zeros: List[int] = [0 for _ in range(batch_size - len(inputs_batch))]
                inputs_batch += zeros
                labels_batch += zeros

            # Convert to numpy arrays of required shape.
            inputs_batches.append(np.array(inputs_batch))
            labels_batches.append(np.array(labels_batch).reshape((batch_size, 1)))

        return inputs_batches, labels_batches

    def _train(
            self,
            x_batches: List[np.ndarray],
            y_batches: List[np.ndarray],
            epochs: int
    ):
        """
        Train the model on batches of inputs and labels.

        Parameters
        ----------
        x_batches: input arrays of shape (<batch_size>, )
        y_batches: label arrays of shape (<batch_size>, 1)
        epochs: number of epochs to train
        """
        # Parameters.
        batch_size: int = x_batches[0].shape[0]
        vocabulary_size: int = len(self.vocabulary)
        embedding_size: int = self.embedding_size

        # Embedding matrix.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        # Initial weights for noise-contrastive estimation.
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Placeholders for inputs and labels.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Look up embeddings.
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Compute the NCE loss, using a sample of the negative labels each time.
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=self.negative_samples,
                           num_classes=vocabulary_size))

        # Optimizer.
        optimizer = tf.train.AdamOptimizer().minimize(loss)

        # Global variables initializer.
        global_init = tf.global_variables_initializer()

        # Train the model.
        with tf.Session() as session:
            session.run(global_init)
            cur_loss = 100.0

            for epoch in range(epochs):
                desc = "Epoch %04d" % (epoch + 1)
                with tqdm.tqdm(desc=desc, total=len(x_batches), postfix={"loss": cur_loss}) as t:
                    for inputs, labels in zip(x_batches, y_batches):
                        feed_dict = {train_inputs: inputs, train_labels: labels}
                        _, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)
                        t.set_postfix({"loss": cur_loss})
                        t.update()

            logger.info("Loss after %04d epochs: %f" % (epochs, cur_loss))

            self.embeddings = embeddings.eval()

    def fit(
            self,
            texts: Iterable[str],
            epochs: int = 1,
            batch_size: int = 128
    ):
        """
        Train the model on a corpus of texts.

        Parameters
        ----------
        texts: collection of texts
        epochs: number of epochs to train
        batch_size: batch size
        """
        logger.info("Creating vocabulary...")
        transformed_texts: List[List[int]] = self.vocabulary.fit_transform(texts)
        logger.info("Successfully created vocabulary with %d words!" % len(self.vocabulary))

        logger.info("Generating training set...")
        inputs, labels = self._generate_training_set(transformed_texts)
        inputs_batches, labels_batches = self._generate_batches(inputs, labels, batch_size)
        logger.info("Successfully generated a training set with %d samples!" % len(inputs))

        logger.info("Training model...")
        self._train(inputs_batches, labels_batches, epochs=epochs)
        logger.info("Successfully trained model!")
