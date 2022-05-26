# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import MISSING


def _default_split_key_mapping():
    return {x: x for x in ["train", "validation", "test"]}


@dataclass
class DatasetInfo:
    key: str = MISSING


@dataclass
class HFDatasetInfo(DatasetInfo):
    key: str = MISSING
    subset: Optional[str] = None
    remove_columns: Optional[List[str]] = None
    # Any is actually list of pairs for renaming the column A to B
    # limited to Any because of OmegaConf limitations
    rename_columns: Optional[List[Any]] = None
    # TODO: Look if we can add text column option and encode transform settings here.
    split_key_mapping: Optional[Dict[str, str]] = field(
        default_factory=_default_split_key_mapping
    )
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TorchVisionDatasetInfo(DatasetInfo):
    key: str = MISSING
    train_split: str = "train"
    val_split: str = "val"
    has_val: bool = True
    test_split: str = "test"


@dataclass
class TrainingSingleDatasetInfo:
    train: List[DatasetInfo] = field(default_factory=lambda: [HFDatasetInfo()])
    val: Optional[List[DatasetInfo]] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    allow_uneven_batches: bool = False
    datamodule_extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingYFCCDatasetInfo:
    metadata_path: str = None
    image_root: str = None
    train_data_fraction: Optional[float] = 0.99
    data_split_random_seed: Optional[int] = 123
    itm_probability: Optional[float] = 0.1
    mlm_probability: Optional[float] = 0.15
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    allow_uneven_batches: bool = False

@dataclass
class ImageFolderInfo:
    train_root: str
    val_root: str
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    allow_uneven_batches: bool = False

@dataclass
class TrainingDatasetsInfo:
    selected: List[str] = field(default_factory=lambda: ["image", "text", "vl"])
    image: Optional[ImageFolderInfo] = None
    text: Optional[TrainingSingleDatasetInfo] = None
    vl: Optional[TrainingYFCCDatasetInfo] = None
    num_classes: int = MISSING


@dataclass
class TrainingArguments:
    # Any lightning args to be pushed here
    lightning: Dict[str, Any] = field(default=dict)
    seed: int = -1
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 0.0002
    adam_eps: float = 1e-08
    adam_weight_decay: float = 0.01
    adam_betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    warmup_steps: int = 2000
    prev_v_num: int = 0
    save_every_min: int = 60
    metrics: List[str] = field(default_factory=lambda: ['accuracy']) 


@dataclass
class ModelArguments:
    pretrained_model_key: Optional[str] = None


@dataclass
class FLAVAArguments:
    datasets: TrainingDatasetsInfo = TrainingDatasetsInfo()
    training: TrainingArguments = TrainingArguments()
    model: ModelArguments = ModelArguments()
