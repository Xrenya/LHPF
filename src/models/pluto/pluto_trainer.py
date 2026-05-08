import logging
import os
from typing import Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torchmetrics import MetricCollection
from src.metrics import MR, minADE, minFDE
from src.metrics.prediction_avg_ade import PredAvgADE
from src.metrics.prediction_avg_fde import PredAvgFDE
from src.optim.warmup_cos_lr import WarmupCosLR

from .loss.esdf_collision_loss import ESDFCollisionLoss

logger = logging.getLogger(__name__)


class LightningTrainer(pl.LightningModule):
    def __init__(
        self,
        model: TorchModuleWrapper,
        lr,
        weight_decay,
        epochs,
        warmup_epochs,
        use_collision_loss=True,
        use_contrast_loss=False,
        regulate_yaw=False,
        comfort_loss_weight=0.0,
        comfort_dt=0.1,
        pretrained_pluto_checkpoint=None,
        objective_aggregate_mode: str = "mean",
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.objective_aggregate_mode = objective_aggregate_mode
        self.history_steps = model.history_steps
        self.use_collision_loss = use_collision_loss
        self.use_contrast_loss = use_contrast_loss
        self.regulate_yaw = regulate_yaw
        self.comfort_loss_weight = comfort_loss_weight
        self.comfort_dt = comfort_dt
        self.pretrained_pluto_checkpoint = pretrained_pluto_checkpoint
        self._pretrained_pluto_loaded = False
        self.metrics = None

        self.radius = model.radius
        self.num_modes = model.num_modes
        self.mode_interval = self.radius / self.num_modes
        self.max_abs_mag_jerk = 8.37
        self.max_abs_lat_accel = 4.89
        self.max_lon_accel = 2.40
        self.min_lon_accel = -4.05
        self.max_abs_yaw_accel = 1.93
        self.max_abs_lon_jerk = 4.13
        self.max_abs_yaw_rate = 0.95

        if use_collision_loss:
            self.collision_loss = ESDFCollisionLoss()

    def on_fit_start(self) -> None:
        self._setup_runtime_state()

    def on_validation_start(self) -> None:
        self._setup_runtime_state()

    def on_test_start(self) -> None:
        self._setup_runtime_state()

    def _setup_runtime_state(self) -> None:
        self._load_pretrained_pluto()
        if self.metrics is not None:
            return
        metrics_collection = MetricCollection(
            [
                minADE().to(self.device),
                minFDE().to(self.device),
                MR(miss_threshold=2).to(self.device),
                PredAvgADE().to(self.device),
                PredAvgFDE().to(self.device),
            ]
        )
        self.metrics = {
            "train": metrics_collection.clone(prefix="train/"),
            "val": metrics_collection.clone(prefix="val/"),
            "test": metrics_collection.clone(prefix="test/"),
        }

    def _load_pretrained_pluto(self) -> None:
        if self.pretrained_pluto_checkpoint is None:
            return
        if self._pretrained_pluto_loaded:
            return

        ckpt = torch.load(
            self.pretrained_pluto_checkpoint,
            map_location=torch.device("cpu"),
        )
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {
            k.replace("model.", "", 1): v
            for k, v in state_dict.items()
        }
        missing_keys, unexpected_keys = self.model.load_state_dict(
            state_dict,
            strict=False,
        )
        logger.info(
            "Loaded pretrained Pluto checkpoint from %s with %d missing and %d "
            "unexpected keys.",
            self.pretrained_pluto_checkpoint,
            len(missing_keys),
            len(unexpected_keys),
        )
        self._pretrained_pluto_loaded = True

    def _step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], prefix: str
    ) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        features, targets, scenarios = batch
        data = features["feature"].data
        self._attach_lhpf_training_history(data)
        res = self.forward(data)

        losses = self._compute_objectives(res, data)
        metrics = self._compute_metrics(res, data, prefix)
        self._log_step(losses["loss"], losses, metrics, prefix)

        return losses["loss"] if self.training else 0.0

    def _attach_lhpf_training_history(self, data) -> None:
        if not getattr(self.model, "use_lhpf", False):
            return
        if "reference_line" not in data:
            raise ValueError(
                "LHPF training requires reference_line features; set "
                "model.feature_builder.build_reference_line=true."
            )
        if "historical_feature" not in data:
            raise ValueError(
                "LHPF training batches must include historical_feature from the "
                "datamodule previous-step pipeline."
            )

        historical_data = data["historical_feature"]
        model_was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                historical_out = self.model(historical_data)
        finally:
            self.model.train(model_was_training)

        historical_embedding = historical_out.get("planning_embedding")
        if historical_embedding is None:
            return

        historical_embedding = historical_embedding.detach()
        historical_valid_mask = self._historical_reference_line_mask(
            historical_data,
            historical_embedding,
        )
        current_batch_size = data["agent"]["position"].shape[0]
        historical_embedding, historical_valid_mask = self._repeat_history_to_batch(
            historical_embedding,
            historical_valid_mask,
            current_batch_size,
        )

        data["reference_line"]["historical_planning_embedding"] = historical_embedding
        data["reference_line"]["historical_planning_valid_mask"] = historical_valid_mask

    def _historical_reference_line_mask(self, historical_data, historical_embedding):
        reference_line = historical_data.get("reference_line")
        if reference_line is None or "valid_mask" not in reference_line:
            bs, num_ref = historical_embedding.shape[:2]
            return torch.ones(
                bs,
                num_ref,
                device=historical_embedding.device,
                dtype=torch.bool,
            )
        return reference_line["valid_mask"].any(-1).to(
            device=historical_embedding.device,
            dtype=torch.bool,
        )

    @staticmethod
    def _repeat_history_to_batch(
        historical_embedding,
        historical_valid_mask,
        current_batch_size,
    ):
        history_batch_size = historical_embedding.shape[0]
        if history_batch_size == current_batch_size:
            return historical_embedding, historical_valid_mask

        if current_batch_size % history_batch_size != 0:
            raise ValueError(
                "Cannot align historical planning embedding batch size "
                f"{history_batch_size} to current batch size {current_batch_size}."
            )

        repeat_count = current_batch_size // history_batch_size
        repeat_dims = (repeat_count,) + (1,) * (historical_embedding.dim() - 1)
        historical_embedding = historical_embedding.repeat(repeat_dims)
        historical_valid_mask = historical_valid_mask.repeat(repeat_count, 1)
        return historical_embedding, historical_valid_mask

    def _compute_objectives(self, res, data) -> Dict[str, torch.Tensor]:
        bs, _, T, _ = res["prediction"].shape

        if self.use_contrast_loss:
            train_num = (bs // 3) * 2 if self.training else bs
        else:
            train_num = bs

        trajectory, probability, prediction = (
            res["trajectory"][:train_num],
            res["probability"][:train_num],
            res["prediction"][:train_num],
        )
        ref_free_trajectory = res.get("ref_free_trajectory", None)

        targets_pos = data["agent"]["target"][:train_num]
        valid_mask = data["agent"]["valid_mask"][:train_num, :, -T:]
        targets_vel = data["agent"]["velocity"][:train_num, :, -T:]

        target = torch.cat(
            [
                targets_pos[..., :2],
                torch.stack(
                    [targets_pos[..., 2].cos(), targets_pos[..., 2].sin()], dim=-1
                ),
                targets_vel,
            ],
            dim=-1,
        )

        # planning loss
        (
            ego_reg_loss,
            ego_cls_loss,
            collision_loss,
            best_trajectory,
        ) = self.get_planning_loss(
            data,
            trajectory,
            probability,
            valid_mask[:, 0],
            target[:, 0],
            train_num,
        )
        if ref_free_trajectory is not None:
            ego_ref_free_reg_loss = F.smooth_l1_loss(
                ref_free_trajectory[:train_num],
                target[:, 0, :, : ref_free_trajectory.shape[-1]],
                reduction="none",
            ).sum(-1)
            ego_ref_free_reg_loss = (
                ego_ref_free_reg_loss * valid_mask[:, 0]
            ).sum() / valid_mask[:, 0].sum()
        else:
            ego_ref_free_reg_loss = ego_reg_loss.new_zeros(1)

        # prediction loss
        prediction_loss = self.get_prediction_loss(
            data, prediction, valid_mask[:, 1:], target[:, 1:]
        )

        if self.training and self.use_contrast_loss:
            contrastive_loss = self._compute_contrastive_loss(
                res["hidden"], data["data_n_valid_mask"]
            )
        else:
            contrastive_loss = prediction_loss.new_zeros(1)

        if self.comfort_loss_weight > 0:
            comfort_loss = self.get_comfort_loss(
                best_trajectory,
                valid_mask[:, 0],
            )
        else:
            comfort_loss = prediction_loss.new_zeros(1)

        loss = (
            ego_reg_loss
            + ego_cls_loss
            + prediction_loss
            + contrastive_loss
            + collision_loss
            + ego_ref_free_reg_loss
            + self.comfort_loss_weight * comfort_loss
        )

        return {
            "loss": loss,
            "reg_loss": ego_reg_loss.item(),
            "cls_loss": ego_cls_loss.item(),
            "ref_free_reg_loss": ego_ref_free_reg_loss.item(),
            "collision_loss": collision_loss.item(),
            "prediction_loss": prediction_loss.item(),
            "contrastive_loss": contrastive_loss.item(),
            "comfort_loss": comfort_loss.item(),
        }

    def get_prediction_loss(self, data, prediction, valid_mask, target):
        """
        prediction: (bs, A-1, T, 6)
        valid_mask: (bs, A-1, T)
        target: (bs, A-1, 6)
        """
        prediction_loss = F.smooth_l1_loss(
            prediction[valid_mask], target[valid_mask], reduction="none"
        ).sum(-1)
        prediction_loss = prediction_loss.sum() / valid_mask.sum()

        return prediction_loss

    def get_planning_loss(self, data, trajectory, probability, valid_mask, target, bs):
        """
        trajectory: (bs, R, M, T, 4)
        valid_mask: (bs, T)
        """
        num_valid_points = valid_mask.sum(-1)
        endpoint_index = (num_valid_points / 10).long().clamp_(min=0, max=7)  # max 8s
        r_padding_mask = ~data["reference_line"]["valid_mask"][:bs].any(-1)  # (bs, R)
        future_projection = data["reference_line"]["future_projection"][:bs][
            torch.arange(bs), :, endpoint_index
        ]

        target_r_index = torch.argmin(
            future_projection[..., 1] + 1e6 * r_padding_mask, dim=-1
        )
        target_m_index = (
            future_projection[torch.arange(bs), target_r_index, 0] / self.mode_interval
        ).long()
        target_m_index.clamp_(min=0, max=self.num_modes - 1)

        target_label = torch.zeros_like(probability)
        target_label[torch.arange(bs), target_r_index, target_m_index] = 1

        best_trajectory = trajectory[torch.arange(bs), target_r_index, target_m_index]

        if self.use_collision_loss:
            collision_loss = self.collision_loss(
                best_trajectory, data["cost_maps"][:bs, :, :, 0].float()
            )
        else:
            collision_loss = trajectory.new_zeros(1)

        reg_loss = F.smooth_l1_loss(best_trajectory, target, reduction="none").sum(-1)
        reg_loss = (reg_loss * valid_mask).sum() / valid_mask.sum()

        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        cls_loss = F.cross_entropy(
            probability.reshape(bs, -1), target_label.reshape(bs, -1).detach()
        )

        if self.regulate_yaw:
            heading_vec_norm = torch.norm(best_trajectory[..., 2:4], dim=-1)
            yaw_regularization_loss = F.l1_loss(
                heading_vec_norm, heading_vec_norm.new_ones(heading_vec_norm.shape)
            )
            reg_loss += yaw_regularization_loss

        return reg_loss, cls_loss, collision_loss, best_trajectory

    def get_comfort_loss(self, trajectory, valid_mask):
        """
        trajectory: (bs, T, 6), ordered as x, y, cos(yaw), sin(yaw), vx, vy
        valid_mask: (bs, T)
        """
        if trajectory.shape[1] < 3:
            return trajectory.new_zeros(1)

        dt = self.comfort_dt
        velocity = trajectory[..., 4:6]
        heading = torch.atan2(trajectory[..., 3], trajectory[..., 2])
        heading_vec = F.normalize(trajectory[..., 2:4], dim=-1)

        acceleration = (velocity[:, 1:] - velocity[:, :-1]) / dt
        acceleration_valid = valid_mask[:, 1:] & valid_mask[:, :-1]
        forward_vec = heading_vec[:, 1:]
        lateral_vec = torch.stack([-forward_vec[..., 1], forward_vec[..., 0]], dim=-1)
        lon_acceleration = (acceleration * forward_vec).sum(-1)
        lat_acceleration = (acceleration * lateral_vec).sum(-1)

        yaw_rate = self._angle_difference(heading[:, 1:], heading[:, :-1]) / dt

        loss = self._comfort_bound_loss(
            lon_acceleration,
            acceleration_valid,
            lower=self.min_lon_accel,
            upper=self.max_lon_accel,
        )
        loss = loss + self._comfort_bound_loss(
            lat_acceleration.abs(),
            acceleration_valid,
            upper=self.max_abs_lat_accel,
        )
        loss = loss + self._comfort_bound_loss(
            yaw_rate.abs(),
            acceleration_valid,
            upper=self.max_abs_yaw_rate,
        )

        if trajectory.shape[1] < 4:
            return loss

        jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / dt
        jerk_valid = acceleration_valid[:, 1:] & acceleration_valid[:, :-1]
        jerk_mag = torch.linalg.norm(jerk, dim=-1)
        jerk_forward_vec = heading_vec[:, 2:]
        lon_jerk = (jerk * jerk_forward_vec).sum(-1)
        yaw_acceleration = (yaw_rate[:, 1:] - yaw_rate[:, :-1]) / dt

        loss = loss + self._comfort_bound_loss(
            jerk_mag,
            jerk_valid,
            upper=self.max_abs_mag_jerk,
        )
        loss = loss + self._comfort_bound_loss(
            lon_jerk.abs(),
            jerk_valid,
            upper=self.max_abs_lon_jerk,
        )
        loss = loss + self._comfort_bound_loss(
            yaw_acceleration.abs(),
            jerk_valid,
            upper=self.max_abs_yaw_accel,
        )

        return loss

    def _comfort_bound_loss(self, values, valid_mask, lower=None, upper=None):
        loss = values.new_zeros(values.shape)
        if lower is not None:
            loss = loss + F.relu(lower - values).square()
        if upper is not None:
            loss = loss + F.relu(values - upper).square()

        valid_mask = valid_mask.to(dtype=values.dtype)
        return (loss * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    @staticmethod
    def _angle_difference(current, previous):
        difference = current - previous
        return torch.atan2(torch.sin(difference), torch.cos(difference))

    def _compute_contrastive_loss(
        self, hidden, valid_mask, normalize=True, tempreture=0.1
    ):
        """
        Compute triplet loss

        Args:
            hidden: (3*bs, D)
        """
        if normalize:
            hidden = F.normalize(hidden, dim=1, p=2)

        if not valid_mask.any():
            return hidden.new_zeros(1)

        x_a, x_p, x_n = hidden.chunk(3, dim=0)

        x_a = x_a[valid_mask]
        x_p = x_p[valid_mask]
        x_n = x_n[valid_mask]

        logits_ap = (x_a * x_p).sum(dim=1) / tempreture
        logits_an = (x_a * x_n).sum(dim=1) / tempreture
        labels = x_a.new_zeros(x_a.size(0)).long()

        triplet_contrastive_loss = F.cross_entropy(
            torch.stack([logits_ap, logits_an], dim=1), labels
        )
        return triplet_contrastive_loss

    def _compute_metrics(self, res, data, prefix) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        if self.metrics is None:
            self._setup_runtime_state()

        # get top 6 modes
        trajectory, probability = res["trajectory"], res["probability"]
        r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
        probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

        bs, R, M, T, _ = trajectory.shape
        trajectory = trajectory.reshape(bs, R * M, T, -1)
        probability = probability.reshape(bs, R * M)
        top_k_prob, top_k_index = probability.topk(6, dim=-1)
        top_k_traj = trajectory[torch.arange(bs)[:, None], top_k_index]

        outputs = {
            "trajectory": top_k_traj[..., :2],
            "probability": top_k_prob,
            "prediction": res["prediction"][..., :2],
            "prediction_target": data["agent"]["target"][:, 1:],
            "valid_mask": data["agent"]["valid_mask"][:, 1:, self.history_steps :],
        }
        target = data["agent"]["target"][:, 0]

        metrics = self.metrics[prefix](outputs, target)
        return metrics

    def _log_step(
        self,
        loss,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = "loss",
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(
            f"loss/{prefix}_{loss_name}",
            loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True if prefix == "train" else False,
        )

        for key, value in objectives.items():
            self.log(
                f"objectives/{prefix}_{key}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        if metrics is not None:
            self.log_dict(
                metrics,
                prog_bar=(prefix == "val"),
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def training_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "train")

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "val")

    def test_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        return self._step(batch, "test")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        # Get optimizer
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        # Get lr_scheduler
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )

        return [optimizer], [scheduler]

    # def on_before_optimizer_step(self, optimizer) -> None:
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print("unused param", name)
