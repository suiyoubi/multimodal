# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from definitions import DatasetInfo, FLAVAArguments, ImageFolderInfo, TrainingArguments, TrainingYFCCDatasetInfo
from hydra.utils import instantiate
from omegaconf import OmegaConf


def build_datamodule_kwargs(dm_config: DatasetInfo, training_config: TrainingArguments):
    return {
        "train_infos": dm_config.train,
        "val_infos": dm_config.val,
        "batch_size": dm_config.batch_size or training_config.batch_size,
        "num_workers": dm_config.num_workers or training_config.num_workers,
        "allow_uneven_batches": dm_config.allow_uneven_batches,
    }

def build_imagenet_datamodule_kwargs(if_config: ImageFolderInfo, training_config: TrainingArguments):
    return {
        "train_root": if_config.train_root,
        "val_root": if_config.val_root,
        "batch_size": if_config.batch_size or training_config.batch_size,
        "num_workers": if_config.num_workers or training_config.num_workers,
        "allow_uneven_batches": if_config.allow_uneven_batches,
    }

def build_yfcc_datamodule_kwargs(yfcc_config: TrainingYFCCDatasetInfo, training_config: TrainingArguments):
    return {
        "metadata_path": yfcc_config.metadata_path,
        "image_root": yfcc_config.image_root,
        "train_data_fraction": yfcc_config.train_data_fraction,
        "data_split_random_seed": yfcc_config.data_split_random_seed,
        "itm_probability": yfcc_config.itm_probability,
        "mlm_probability": yfcc_config.mlm_probability,
        "batch_size": yfcc_config.batch_size or training_config.batch_size,
        "num_workers": yfcc_config.num_workers or training_config.num_workers,
        "allow_uneven_batches": yfcc_config.allow_uneven_batches,
    }

def build_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running FLAVA"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    cli_conf.pop("config")
    config: FLAVAArguments = OmegaConf.merge(conf, cli_conf)

    assert (
        "max_steps" in config.training.lightning
    ), "lightning config must specify 'max_steps'"

    return config
