# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset builders."""
import os
import abc
import dataclasses
import functools
from typing import Optional, Sequence, Tuple

import cachetools
from google.protobuf import text_format
import jax
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

AbstractSplit = tfds.core.splits.AbstractSplit
ReadInstruction = tfds.core.ReadInstruction


class DatasetBuilder(abc.ABC):
  """Abstract dataset builder."""

  @property
  @abc.abstractmethod
  def num_examples(self) -> int:
    """Total number of examples in the dataset split."""

  @abc.abstractmethod
  def as_dataset(self) -> tf.data.Dataset:
    """Returns a dataset as a tf.data.Dataset object."""

  @abc.abstractmethod
  def get_num_fake_examples(self, batch_size_per_process: int) -> int:
    """Number of fake examples processed in the current process."""


@dataclasses.dataclass
class PotatoCsvBuilder(DatasetBuilder):
  """Dataset builder for Potato Leaf Disease dataset from CSV, preloading all images into RAM."""
  name: str
  split: str
  csv_path: str = ""
  data_dir: str = ""
  shuffle_files: bool = False
  shuffle_seed: Optional[int] = None
  use_cache: bool = True

  def _load_dataframe(self):
    import pandas as pd
    df = pd.read_csv(self.csv_path)
    if 'train' in self.split.lower():
      df = df[df['Type'] == 'Train']
    elif 'val' in self.split.lower():
      df = df[df['Type'].isin(['Validation', 'Val'])]
    elif 'test' in self.split.lower():
      df = df[df['Type'] == 'Test']
    return df.reset_index(drop=True)

  @property
  def num_examples(self) -> int:
    return len(self._load_dataframe())

  def as_dataset(self) -> tf.data.Dataset:
    import numpy as np
    import pandas as pd
    from PIL import Image

    df = self._load_dataframe()
    label_map = {
        'Bacteria': 0,
        'Fungi': 1,
        'Nematode': 2,
        'Pest': 3,
        'Phytopthora': 4,
        'Virus': 5,
    }
    image_paths = [
        os.path.join(self.data_dir, str(row['Label']), str(row['Path']))
        for _, row in df.iterrows()
    ]
    labels = [label_map.get(str(row['Label']), 0) for _, row in df.iterrows()]

    if self.use_cache:
      print(f"Loading {len(image_paths)} images into RAM for split '{self.split}'...")
      cached_images = []
      valid_labels = []
      for path, lbl in zip(image_paths, labels):
        try:
          img = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
          cached_images.append(np.array(img, dtype=np.uint8))
          valid_labels.append(lbl)
        except Exception as e:
          print(f"Error reading image: {path} ({e})")
          continue

      print(f"Successfully loaded {len(cached_images)} images of split '{self.split}' into RAM.")
      ds = tf.data.Dataset.from_tensor_slices({
          'image': np.array(cached_images, dtype=np.uint8),
          'label': np.array(valid_labels, dtype=np.int32),
      })
    else:
      def _read_image(path, label):
        image_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))
        return {'image': tf.cast(img, tf.uint8), 'label': label}

      ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
      ds = ds.map(_read_image, num_parallel_calls=tf.data.AUTOTUNE)

    return ds

  def get_num_fake_examples(self, batch_size_per_process: int) -> int:
    return _get_num_fake_examples(
        jax.process_index(), batch_size_per_process, [self.num_examples]
    )


@dataclasses.dataclass
class TfdsBuilder(DatasetBuilder):
  """Dataset builder for TFDS datasets."""
  name: str
  split: str
  data_dir: Optional[str] = None
  manual_dir: Optional[str] = None
  shuffle_files: bool = False
  shuffle_seed: Optional[int] = None
  try_gcs: bool = False
  ignore_errors: bool = False

  @property
  def num_examples(self) -> int:
    return self._tfds_builder.info.splits[self.split].num_examples

  def as_dataset(self) -> tf.data.Dataset:
    split = tfds.split_for_jax_process(self.split, drop_remainder=False)
    read_config = tfds.ReadConfig(
        shuffle_seed=self.shuffle_seed, skip_prefetch=True, try_autocache=False)
    data = self._tfds_builder.as_dataset(
        split=split,
        decoders={'image': tfds.decode.SkipDecoding()},
        shuffle_files=self.shuffle_files,
        read_config=read_config)
    if self.ignore_errors:
      data = data.ignore_errors()
    return data

  def get_num_fake_examples(self, batch_size_per_process: int) -> int:
    builder = self._tfds_builder
    num_examples = [
        builder.info.splits[split].num_examples for split in tfds.even_splits(
            self.split, jax.process_count(), drop_remainder=False)
    ]
    return _get_num_fake_examples(jax.process_index(), batch_size_per_process,
                                  num_examples)

  @property
  def _tfds_builder(self):
    return _get_tfds_builder(self.name, self.data_dir, self.manual_dir,
                             self.try_gcs)


def _get_num_fake_examples(process_index: int, batch_size_per_process: int,
                           num_examples_per_process: Sequence[int]) -> int:
  """Returns the number of fake examples to use in a given process."""
  assert process_index < len(num_examples_per_process)
  num_examples_max = max(num_examples_per_process)
  num_examples_process = num_examples_per_process[process_index]
  num_fake_examples = num_examples_max - num_examples_process
  num_fake_examples += (-num_examples_max) % batch_size_per_process
  return num_fake_examples


@cachetools.cached(
    cache={},
    key=lambda name, data_dir, *_: cachetools.keys.hashkey(name, data_dir))
def _get_tfds_builder(name, data_dir, manual_dir, try_gcs):
  if 'from_directory:' in name:
    return tfds.builder_from_directory(data_dir)
  data_builder = tfds.builder(name=name, data_dir=data_dir, try_gcs=try_gcs)
  data_builder.download_and_prepare(
      download_config=tfds.download.DownloadConfig(manual_dir=manual_dir))
  return data_builder


def get_dataset_builder(*, name: str, **kwargs) -> DatasetBuilder:
  """Returns the builder to use for a given dataset split."""
  if name.startswith('potato'):
    return PotatoCsvBuilder(name=name, **kwargs)
  return TfdsBuilder(name=name, **kwargs)
