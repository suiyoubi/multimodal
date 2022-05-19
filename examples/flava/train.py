# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from callbacks.multimodal_eval import MultimodalEvalCallback
from data import ImageDataModule, ImageDataModuleOld, MLMDataModule, MultiDataModule, VLDataModule, YFCCDataModule
from definitions import DatasetInfo, FLAVAArguments, TrainingSingleDatasetInfo, HFDatasetInfo
from model import FLAVAPreTrainingLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import build_config, build_datamodule_kwargs, build_imagenet_datamodule_kwargs, build_yfcc_datamodule_kwargs
from datetime import timedelta
import os


def main():
    config: FLAVAArguments = build_config()
    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    datamodules = []
    if "image" in config.datasets.selected:
        imagenet_datamodule = ImageDataModuleOld(
            **build_imagenet_datamodule_kwargs(config.datasets.image, config.training)
        )
        datamodules.append(imagenet_datamodule)
    else:
        imagenet_datamodule = None

    if "text" in config.datasets.selected:
        mlm_datamodule = MLMDataModule(
            **build_datamodule_kwargs(config.datasets.text, config.training)
        )
        datamodules.append(mlm_datamodule)

    if "vl" in config.datasets.selected:
        # We only have YFCC dataset on draco, use HFDataset for other usage
        if os.path.isfile(config.datasets.vl.metadata_path):
            vl_datamodule = YFCCDataModule(
                **build_yfcc_datamodule_kwargs(config.datasets.vl, config.training)
            )
        else:
            print(f'{config.datasets.vl.metadata_path} not exists. Building VL DataModule based on HF for testing')
            vl_datamodule = VLDataModule(
                **build_datamodule_kwargs(
                    TrainingSingleDatasetInfo(
                        train=[HFDatasetInfo(
                            key='red_caps',
                            subset='jellyfish',
                            rename_columns=[['caption', 'text']]
                        )],
                        val=[HFDatasetInfo(
                            key='red_caps',
                            subset='jellyfish',
                            rename_columns=[['caption', 'text']],
                            split_key_mapping={'validation': 'train'}
                        )]
                    ),
                    config.training
                )
            )
        datamodules.append(vl_datamodule)

    datamodule = MultiDataModule(datamodules)

    datamodule.setup("fit")
    model = FLAVAPreTrainingLightningModule(
        learning_rate=config.training.learning_rate,
        adam_eps=config.training.adam_eps,
        adam_weight_decay=config.training.adam_weight_decay,
        adam_betas=config.training.adam_betas,
        warmup_steps=config.training.warmup_steps,
        max_steps=config.training.lightning.max_steps,
        **config.model,
    )

    callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename="{epoch}-{step}",
                train_time_interval=timedelta(minutes=config.training.save_every_min),
                save_last=True,
                save_top_k = -1
            )
    ]
    if imagenet_datamodule:
        callbacks.append(MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule))
    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=callbacks,
        strategy="ddp",
    )

    prev_ckpt = os.path.join(
        config.training.lightning['default_root_dir'], 
        'lightning_logs', 
        f'version_{config.training.prev_v_num}', 
        'checkpoints', 
        'last.ckpt'
        )
    if os.path.exists(prev_ckpt):
        print(f'Resuming from last checkpoint: {prev_ckpt}')
    else:
        prev_ckpt = None

    trainer.fit(
        model, datamodule=datamodule,
        ckpt_path=prev_ckpt
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
