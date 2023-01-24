import torch
import pickle
import argparse
import os
import math
import numpy as np
import pytorch_lightning as pl
from functools import partial
from graph_tool import load_graph

from ..models import TransE
from ..dataloader import TripletDataset, generate_corruption_eval, generate_corruption_fit, DataMapper
from ..loss import PairwiseLoss
from ..evaluation import hits_at_n_score, mrr_score, train_test_split

SCORE_COMPARISSON_PRECISION = 1e5

def compare(score_corr: torch.Tensor, score_pos: torch.Tensor, comparisson_type: str = 'best'):
    """
    Compares the scores of corruptions and positives using the specified strategy.

    :param score_corr: torch.Tensor of scores of corruptions.
    :param score_pos: torch.Tensor of scores of positives.
    :param comparisson_type: Comparisson strategy.
    :return:
    """

    assert comparisson_type in ['worst', 'best', 'middle'], 'Invalid score comparisson type!'

    score_pos = (score_pos * SCORE_COMPARISSON_PRECISION).int()
    score_corr = (score_corr * SCORE_COMPARISSON_PRECISION).int()

    if comparisson_type == 'best':
        return (score_corr > score_pos).int().sum()
    elif comparisson_type == 'middle':
        return (score_corr > score_pos).int().sum() + ((score_corr == score_pos).int().sum() / 2).ceil().int()
    else:
        return (score_corr >= score_pos).int().sum()


def val_collator_upd(triplets, collate_fn=None, eta_val=3, **kwargs):
    """
    Generates entities_for_corrupt.

    :param triplets: torch.Tensor of shape (n, 3) of triplets.
    :param collate_fn: generate corruptions collate fn.
    :param eta_val: Number of corruptions per positive.
    :param kwargs: key-word args given to collate_fn.
    :return: List of (positive, corruptions)
    """

    entities = torch.unique(torch.cat([triplets[:, 0], triplets[:, 2]], dim=0))
    collations = []
    for triplet in triplets:
        kwargs['entities_for_corrupt'] = entities[torch.randperm(entities.shape[0])[:eta_val]]
        collations.append((triplet, collate_fn(triplets, **kwargs)))
    return collations


class LitModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.model = TransE()
        self.mapper = None

        self.criterion = PairwiseLoss(margin=1.0)

        self.train_dataset = None
        self.train_bs = self.hparams.train_bs
        self.val_dataset = None
        self.val_bs = self.hparams.val_bs
        self.total_steps = None

        self.train_collate = None
        self.val_collate = None

        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx=None):
        pos, neg = batch

        pos_scores = self.model(pos[:, 0], pos[:, 1], pos[:, 2])
        neg_scores = self.model(neg[:, 0], neg[:, 1], neg[:, 2])

        loss = self.criterion(pos_scores, neg_scores)

        self.log('loss', loss.view(1, 1))

        return loss

    def validation_step(self, batch, batch_idx):
        ranks = []
        for pos_trip, neg_trip in batch:
            with torch.set_grad_enabled(False):
                score_pos = self.model(pos_trip[:, 0], pos_trip[:, 1], pos_trip[:, 2])
                score_corr = self.model(neg_trip[:, 0], neg_trip[:, 1], neg_trip[:, 2])

            if self.hpatams.val_corrupt == 's+o':
                obj_corr_score = score_corr[:score_corr.shape[0] // 2]
                subj_corr_score = score_corr[score_corr.shape[0] // 2:]
                rank = torch.cat([
                    compare(subj_corr_score, score_pos, 'best') + 1,
                    compare(obj_corr_score, score_pos, 'best') + 1
                ], dim=0)
            else:
                rank = compare(score_corr, score_pos, 'best')  + 1
            ranks.append(rank)

        return {'rank' : torch.cat(ranks, dim=0)}

    def train_epoch_end(self, outputs):
        loss = torch.cat([x['loss'] for x in outputs])

        self.log('epoch_loss', loss.mean())

    def validation_epoch_end(self, outputs):
        ranks = torch.cat([x['rank'] for x in outputs], dim=0) # should be in outputs

        hits_10 = hits_at_n_score(ranks, 10)
        hits_3 = hits_at_n_score(ranks, 3)
        hits_1 = hits_at_n_score(ranks, 1)
        mrr = mrr_score(ranks)

        self.log("hits@10", hits_10.item())
        self.log("hits@3", hits_3.item())
        self.log("hits@1", hits_1.item())
        self.log("MRR", mrr.item())


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.98), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            epochs=self.hparams.epochs,
            total_steps=self.total_steps,
            max_lr=self.hparams.lr,
            anneal_strategy=self.hparams.anneal_strategy,
            pct_start=self.hparams.pct_start
        )
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : lr_scheduler
        }

    def setup(self, stage=None):
        with open(self.hparams.train_dataset, 'rb') as f:
            triplets = pickle.load(f)

        self.mapper = DataMapper(triplets)
        self.train_dataset = self.mapper.transform(triplets, return_tensors='pt')
        self.train_dataset, self.val_dataset = train_test_split(self.train_dataset, test_size=self.hparams.val_ratio)

        self.train_dataset = TripletDataset(self.train_dataset[:, 0], self.train_dataset[:, 1],
                                            self.train_dataset[:, 2])

        self.val_dataset = TripletDataset(self.val_dataset[:, 0], self.val_dataset[:, 1], self.val_dataset[:, 2])

        self.total_steps = math.ceil(len(self.train_dataset) * self.hparams.epochs /
                                     (self.train_bs * self.hparams.accumulate_grad_batches * self.gpus))

        self.train_collate = partial(generate_corruption_fit, eta=self.hparams.eta, entities_list=None, ent_size=0,
                                     corrupt=self.hparams.train_corrupt)
        self.val_collate = partial(val_collator_upd, collate_fn=generate_corruption_eval,
                                   corrupt=self.hparams.val_corrupt, eta_val=self.hparams.eta_val)

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.train_bs, shuffle=True,
                                             num_workers=8, pin_memory=True, collate_fn=self.train_collate)

        print("Train loader is OK!")
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.val_bs, shuffle=True,
                                             num_workers=8, pin_memory=True, collate_fn=self.val_collate)

        print("Val loader is OK!")
        return loader


def configure_options():
    parser = argparse.ArgumentParser(description="Process arguements for training NN's.")

    parser.add_argument("--eta", type=int, default=1, help="Number of negative per 1 triplet in train.")
    parser.add_argument("--eta_val", type=int, default=3, help="Number of negative per 1 triplet in evaluation.")
    parser.add_argument("--val_ratio", type=float, default=3, help="Ratio of val split of train.")
    parser.add_argument("--train_corrupt", type=str, default='s+o', help="Which part of triplet to corrupt during "
                                                                         "training,  can be one of ['s+o', 'o', 's']")
    parser.add_argument("--val_corrupt", type=str, default='s+o', help="Which part of triplet to corrupt during "
                                                                       "validation,  can be one of ['s+o', 'o', 's']")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = configure_options()

    model = LitModel(params)
    model.setup()

    logger = pl.loggers.TensorboardLogger(params.logdir)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=params.save_path + f'mode-v_{logger.version}',
            filename='{epoch}-{val_loss:.3f}',
            every_n_epoch=1,
            save_top_k=1,
            mode='min',
            monitor='val_loss'
        )
    ]

    trainer = pl.Trainer(
        accumulate_grad_batches=params.accumulate_grad_batches,
        startegy=params.startegy,
        log_every_n_steps=100,
        val_check_interval=1.0,
        accelerator='gpu',
        max_epochs=params.epochs,
        devices=params.gpus,
        gradient_clip_val=10.0,
        gradient_clip_algorithm='norm',
        precision="bf16",
        logger=logger,
        callbakcs=callbacks
    )

    trainer.validate(model)

    trainer.fit(model)
