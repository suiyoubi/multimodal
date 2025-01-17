# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from common.data import MultiDataModule
from flava.callbacks.multimodal_eval import MultimodalEvalCallback
from flava.data import ImageDataModuleOld, MLMDataModule, VLDataModule, YFCCDataModule
from flava.definitions import DatasetInfo, FLAVAArguments, TrainingSingleDatasetInfo, HFDatasetInfo
from flava.model import FLAVAPreTrainingLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from flava.utils import build_config, build_datamodule_kwargs, build_imagenet_datamodule_kwargs, build_yfcc_datamodule_kwargs
from datetime import timedelta
import os
import torch

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
            print('Building VL DataModule based on HF for testing')
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

    # Resume from the latest run
    # Try to find the latest version num to resume on in case any intermediate run fails
    lightning_logs_dir = os.path.join(config.training.lightning['default_root_dir'], 'lightning_logs')
    prev_ckpt = None
    if os.path.exists(lightning_logs_dir):
        latest_num = -1
        for version_str in sorted(os.listdir(lightning_logs_dir)):
            version_num = int(version_str.replace('version_', ''))
            ckpt_dir = os.path.join(lightning_logs_dir, version_str, 'checkpoints', 'last.ckpt')
            if os.path.exists(ckpt_dir) and version_num > latest_num:
                version_num = latest_num
                prev_ckpt = ckpt_dir

    if prev_ckpt is not None and os.path.exists(prev_ckpt):
        print(f'Resuming from last checkpoint: {prev_ckpt}')
        # Known issue with Pytorch-lightning that reset logger step incorrectly 
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/12274
        checkpoint = torch.load(prev_ckpt, map_location='cpu')
        global_step_offset = checkpoint['global_step']
        trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset
        del checkpoint

    trainer.fit(
        model, datamodule=datamodule,
        ckpt_path=prev_ckpt
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
