"""Dataset base class."""

import abc
import functools
import operator
from typing import Callable
import ml_collections

import registry
import tensorflow as tf

DatasetRegistry = registry.Registry()

num_parallel_calls = tf.data.experimental.AUTOTUNE


# num_parallel_calls = None


def mix_datasets(cfg, input_fns, weights):
    """Mix multiple datasets according to weights.

    Args:
      input_fns: a list of input_fn's. Each input_fn takes in an input_context and
          produces a tf.data.Dataset instance.
      weights: a list of floats where weights[i] represents the probability to
          sample from input_fns[i].

    Returns:
      a tf.data.Dataset instance.
    """

    def input_fn(input_context):
        dses = []
        for ifn in input_fns:
            dses.append(ifn(input_context))
        mixed_ds = tf.data.Dataset.sample_from_datasets(dses, weights)
        return mixed_ds

    if cfg.debug:
        return input_fn(input_context=None)

    return tf.distribute.get_strategy().distribute_datasets_from_function(input_fn)


class Dataset(abc.ABC):
    """A dataset that handles creating a tf.data.Dataset."""

    def __init__(self, config: ml_collections.ConfigDict):
        """Constructs the dataset."""
        self.config_all = config
        self.config = config.dataset
        self.task_config = config.task

    @abc.abstractmethod
    def extract(self, example, training):
        """Extracts needed features & annotations into a flat dictionary.

        Note: be consisous about 0 in label, which should probably reserved for
           special use (such as padding).

        Args:
          example: `dict` of raw features.
          training: `bool` of training vs eval mode.

        Returns:
          example: `dict` of relevant features and labels
        """

    @abc.abstractmethod
    def load_dataset(self, input_context, training):
        """Load tf.data.Dataset from sources such as TFDS or TFRecord files."""

    def parse_example(self, example, training):
        del training
        return example

    def filter_example(self, unused_example, unused_training):
        return True

    def pipeline(self,
                 process_single_example: Callable[[tf.data.Dataset, int, bool, bool],
                 tf.data.Dataset],
                 global_batch_size: int,
                 training: bool,
                 validation: bool,
                 ):
        """Data pipeline from name to preprocessed examples.

        Args:
          process_single_example: a function that takes single example dataset and
            returns processed example dataset.
          global_batch_size: global batch size.
          training: training vs eval mode.

        Returns:
          An input_fn which generates a tf.data.Dataset instance.
        """
        config = self.config
        config_all = self.config_all

        def input_fn(input_context):
            dataset = self.load_dataset(input_context, training)

            if input_context:
                batch_size = input_context.get_per_replica_batch_size(global_batch_size)
                # Sharding is not neccesary for TFDS given read_config above.
                # dataset = dataset.shard(input_context.num_input_pipelines,
                #                         input_context.input_pipeline_id)
            else:
                batch_size = global_batch_size

            if config_all.debug != 2:
                if config.cache_dataset:
                    dataset = dataset.cache()

                if training:
                    options = tf.data.Options()
                    options.deterministic = False
                    options.experimental_slack = True
                    dataset = dataset.with_options(options)
                    buffer_size = config.get('buffer_size', 0)
                    if buffer_size <= 0:
                        buffer_size = 10 * batch_size
                    dataset = dataset.shuffle(buffer_size)
                    dataset = dataset.repeat()

                dataset = dataset.map(
                    lambda x: self.parse_example(x, training),
                    num_parallel_calls=num_parallel_calls
                )

                dataset = dataset.filter(
                    lambda x: self.filter_example(x, training)
                )

                dataset = dataset.map(
                    lambda x: self.extract(x, training),
                    num_parallel_calls=num_parallel_calls
                )
            if process_single_example:
                dataset = process_single_example(
                    dataset, config.batch_duplicates, training, validation)

            # TODO(b/181662974): Revert this and support non-even batch sizes.
            # dataset = dataset.batch(batch_size, drop_remainder=training)
            dataset = dataset.padded_batch(batch_size, drop_remainder=training or validation)

            if config_all.debug != 2:
                if config.batch_duplicates > 1 and training:
                    dataset = dataset.map(
                        self._flatten_dims,
                        num_parallel_calls=num_parallel_calls
                    )
                dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return dataset

        return input_fn

    def debug_pipeline(self, x, training):
        x = self.parse_example(x, training)
        x = self.extract(x, training)
        # x = self._flatten_dims(x)
        return x

    def _flatten_dims(self, example):
        """Flatten first 2 dims when batch is independently duplicated."""

        def flatten_first_2_dims(t):
            """Merge first 2 dims."""
            shape_list = t.shape.as_list()
            new_bsz = functools.reduce(operator.mul, shape_list[:2])
            out_shape = [new_bsz] + shape_list[2:]
            return tf.reshape(t, out_shape)

        return tf.nest.map_structure(flatten_first_2_dims, example)

    @property
    @abc.abstractmethod
    def num_train_examples(self):
        """Number of training examples."""

    @property
    @abc.abstractmethod
    def num_eval_examples(self):
        """Number of eval examples."""
