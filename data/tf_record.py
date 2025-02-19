import abc
import ml_collections

import tensorflow as tf

from data import dataset as ds


class TFRecordDataset(ds.Dataset):
    """A dataset created from tfrecord files."""

    def __init__(self, config: ml_collections.ConfigDict):
        """Constructs the dataset."""
        super().__init__(config)
        self.dataset_cls = tf.data.TFRecordDataset

    def load_dataset(self, input_context, training):
        """Load tf.data.Dataset from TFRecord files."""
        if training or self.config.eval_split == 'train':
            file_pattern = self.config.train_file_pattern
        else:
            file_pattern = self.config.eval_file_pattern

        db_type = 'training' if training else 'evaluation'
        print(f'loading {db_type} tfrecord dataset from  {file_pattern}')

        dataset = tf.data.Dataset.list_files(
            file_pattern,
            shuffle=training,
            # shuffle=False,
        )
        dataset = dataset.interleave(
            self.dataset_cls,
            cycle_length=32,
            deterministic=not training,
            num_parallel_calls=ds.num_parallel_calls,
            # deterministic=True,
            # num_parallel_calls=0,
        )
        return dataset

    @abc.abstractmethod
    def get_feature_map(self, training):
        """Returns feature map(s) for parsing the TFExample.

        Returns a single feature map (a dict) to parse a TFEXample.
        Returns a tuple of (context feature map, sequence feature map) to parse a
        TFSequenceExample. Context features are non-sequence features, i.e.
        independent of time/frame. Sequence features have time/frame dimension.

        Args:
          training: `bool` of training vs eval mode.
        """

    def parse_example(self, example, training):
        """Parse the serialized example into a dictionary of tensors.

        Args:
          example: the serialized tf.train.Example or tf.train.SequenceExample.
          training: `bool` of training vs eval mode.

        Returns:
          a dictionary of feature name to tensors.
        """
        feature_map = self.get_feature_map(training)
        if isinstance(feature_map, dict):
            example = tf.io.parse_single_example(example, feature_map)
        else:
            context_features, sequence_features = feature_map
            example, sequence = tf.io.parse_single_sequence_example(
                example, context_features, sequence_features)
            example.update(sequence)

        for k in example:
            if isinstance(example[k], tf.SparseTensor):
                if example[k].dtype == tf.string:
                    example[k] = tf.sparse.to_dense(example[k], default_value='')
                else:
                    example[k] = tf.sparse.to_dense(example[k], default_value=0)
        return example

    @property
    def num_train_examples(self):
        return self.config.train_num_examples

    @property
    def num_eval_examples(self):
        return self.config.eval_num_examples if not self.task_config.get(
            'unbatch', False) else None
