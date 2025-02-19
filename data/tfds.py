import ml_collections

import tensorflow_datasets as tfds


from data import dataset as dataset_lib

class TFDSDataset(dataset_lib.Dataset):
    """A dataset created from a TFDS dataset.

      Each example is a dictionary, but the fields may be different for each
      dataset.

      Each task would have a list of required fields (e.g. bounding boxes for
      object detection). When a dataset is used for a specific task, it should
      contain all the fields required by that task.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        """Constructs the dataset."""
        super().__init__(config)
        self.builder = tfds.builder(self.config.tfds_name,
                                    data_dir=self.config.get('data_dir', None))
        self.builder.download_and_prepare()
        self.allowed_tasks = []

    def load_dataset(self, input_context, training):
        """Load tf.data.Dataset from TFDS."""
        split = self.config.train_split if training else self.config.eval_split
        # For TFDS, pass input_context using read_config to make TFDS read
        # different parts of the dataset on different workers.
        read_config = tfds.ReadConfig(input_context=input_context)
        if isinstance(split, list):
            dataset = self.builder.as_dataset(
                split=split[0], shuffle_files=training, read_config=read_config)
            for i in range(1, len(split)):
                dataset.concatenate(self.builder.as_dataset(
                    split=split[i], shuffle_files=training, read_config=read_config))
        else:
            dataset = self.builder.as_dataset(
                split=split, shuffle_files=training, read_config=read_config)
        return dataset

    @property
    def num_train_examples(self):
        return self.builder.info.splits[self.config.train_split].num_examples

    @property
    def num_eval_examples(self):
        return self.builder.info.splits[
            self.config.eval_split].num_examples if not self.task_config.get(
            'unbatch', False) else None
