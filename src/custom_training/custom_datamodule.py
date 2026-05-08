import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data
from omegaconf import DictConfig
from torch.utils.data.sampler import WeightedRandomSampler

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.data_augmentation.abstract_data_augmentation import (
    AbstractAugmentor,
)
from nuplan.planning.training.data_loader.distributed_sampler_wrapper import (
    DistributedSamplerWrapper,
)
from nuplan.planning.training.data_loader.scenario_dataset import ScenarioDataset
from nuplan.planning.training.data_loader.splitter import AbstractSplitter
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    TargetsType,
    move_features_type_to_device,
)
from nuplan.planning.training.preprocessing.feature_collate import FeatureCollate
from nuplan.planning.training.preprocessing.feature_preprocessor import (
    FeaturePreprocessor,
)
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool

logger = logging.getLogger(__name__)

DataModuleNotSetupError = RuntimeError('Data module has not been setup, call "setup()"')


class IterationOffsetScenario:
    """Scenario view whose iteration 0 is offset into the source scenario."""

    def __init__(self, scenario: AbstractScenario, iteration_offset: int) -> None:
        self._scenario = scenario
        self._iteration_offset = max(0, iteration_offset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scenario, name)

    @property
    def token(self) -> str:
        return f"{self._scenario.token}_iter_{self._iteration_offset:04d}"

    @property
    def scenario_name(self) -> str:
        return f"{self._scenario.scenario_name}_iter_{self._iteration_offset:04d}"

    @property
    def scenario_type(self) -> str:
        return self._scenario.scenario_type

    @property
    def log_name(self) -> str:
        return self._scenario.log_name

    @property
    def map_api(self) -> Any:
        return self._scenario.map_api

    @property
    def database_interval(self) -> float:
        return self._scenario.database_interval

    @property
    def initial_ego_state(self) -> Any:
        return self.get_ego_state_at_iteration(0)

    @property
    def initial_tracked_objects(self) -> Any:
        return self.get_tracked_objects_at_iteration(0)

    @property
    def start_time(self) -> Any:
        return self.get_time_point(0)

    @property
    def end_time(self) -> Any:
        return self._scenario.end_time

    @property
    def ego_vehicle_parameters(self) -> Any:
        return self._scenario.ego_vehicle_parameters

    def _num_iterations(self) -> Optional[int]:
        if not hasattr(self._scenario, "get_number_of_iterations"):
            return None
        return self._scenario.get_number_of_iterations()

    def _shift_iteration(self, iteration: int) -> int:
        shifted_iteration = self._iteration_offset + iteration
        num_iterations = self._num_iterations()
        if num_iterations is None:
            return max(0, shifted_iteration)
        return max(0, min(shifted_iteration, num_iterations - 1))

    def get_number_of_iterations(self) -> int:
        num_iterations = self._num_iterations()
        if num_iterations is None:
            return 1
        return max(1, num_iterations - self._iteration_offset)

    def get_time_point(self, iteration: int) -> Any:
        return self._scenario.get_time_point(self._shift_iteration(iteration))

    def get_ego_state_at_iteration(self, iteration: int) -> Any:
        return self._scenario.get_ego_state_at_iteration(
            self._shift_iteration(iteration)
        )

    def get_tracked_objects_at_iteration(self, iteration: int) -> Any:
        return self._scenario.get_tracked_objects_at_iteration(
            self._shift_iteration(iteration)
        )

    def get_sensors_at_iteration(self, iteration: int) -> Any:
        return self._scenario.get_sensors_at_iteration(self._shift_iteration(iteration))

    def get_lidar_to_ego_transform(self) -> Any:
        return self._scenario.get_lidar_to_ego_transform()

    def get_mission_goal(self) -> Any:
        return self._scenario.get_mission_goal()

    def get_route_roadblock_ids(self) -> List[str]:
        return self._scenario.get_route_roadblock_ids()

    def get_traffic_light_status_at_iteration(self, iteration: int) -> Any:
        return self._scenario.get_traffic_light_status_at_iteration(
            self._shift_iteration(iteration)
        )

    def get_past_timestamps(self, iteration: int, *args: Any, **kwargs: Any) -> Any:
        return self._scenario.get_past_timestamps(
            self._shift_iteration(iteration), *args, **kwargs
        )

    def get_future_timestamps(self, iteration: int, *args: Any, **kwargs: Any) -> Any:
        return self._scenario.get_future_timestamps(
            self._shift_iteration(iteration), *args, **kwargs
        )

    def get_ego_past_trajectory(
        self, iteration: int, *args: Any, **kwargs: Any
    ) -> Any:
        return self._scenario.get_ego_past_trajectory(
            self._shift_iteration(iteration), *args, **kwargs
        )

    def get_ego_future_trajectory(
        self, iteration: int, *args: Any, **kwargs: Any
    ) -> Any:
        return self._scenario.get_ego_future_trajectory(
            self._shift_iteration(iteration), *args, **kwargs
        )

    def get_past_tracked_objects(
        self, iteration: int, *args: Any, **kwargs: Any
    ) -> Any:
        return self._scenario.get_past_tracked_objects(
            self._shift_iteration(iteration), *args, **kwargs
        )

    def get_future_tracked_objects(
        self, iteration: int, *args: Any, **kwargs: Any
    ) -> Any:
        return self._scenario.get_future_tracked_objects(
            self._shift_iteration(iteration), *args, **kwargs
        )


class LHPFScenarioDataset(torch.utils.data.Dataset):
    """
    Dataset that returns the current scenario feature plus a previous-step feature.

    The trainer uses the previous feature to run the frozen/no-loss history pass
    and attach a detached planning embedding before computing the current loss.
    """

    def __init__(
        self,
        scenarios: List[AbstractScenario],
        feature_preprocessor: FeaturePreprocessor,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        current_iteration: int = 1,
        previous_iteration_delta: int = 1,
    ) -> None:
        self._scenarios = scenarios
        self._feature_preprocessor = feature_preprocessor
        self._augmentors = augmentors
        self._current_iteration = max(0, current_iteration)
        self._previous_iteration_delta = max(1, previous_iteration_delta)

    def __len__(self) -> int:
        return len(self._scenarios)

    def __getitem__(
        self, idx: int
    ) -> Tuple[FeaturesType, TargetsType, AbstractScenario]:
        scenario = self._scenarios[idx]
        current_iteration = self._resolve_current_iteration(scenario)
        previous_iteration = max(0, current_iteration - self._previous_iteration_delta)

        current_scenario = IterationOffsetScenario(scenario, current_iteration)
        previous_scenario = IterationOffsetScenario(scenario, previous_iteration)

        features, targets = self._compute_features(current_scenario)
        if self._augmentors is not None:
            for augmentor in self._augmentors:
                features, targets = augmentor.augment(
                    features, targets, current_scenario
                )

        previous_features, _ = self._compute_features(previous_scenario)
        feature_name = next(iter(features))
        previous_feature = previous_features[feature_name]
        features[feature_name].data["historical_feature"] = (
            self._strip_historical_feature(previous_feature.data)
        )

        features = {
            feature_name: feature.to_feature_tensor()
            for feature_name, feature in features.items()
        }
        targets = {
            target_name: target.to_feature_tensor()
            for target_name, target in targets.items()
        }

        return features, targets, current_scenario

    def _resolve_current_iteration(self, scenario: AbstractScenario) -> int:
        if not hasattr(scenario, "get_number_of_iterations"):
            return self._current_iteration
        return min(
            self._current_iteration,
            max(0, scenario.get_number_of_iterations() - 1),
        )

    def _compute_features(
        self, scenario: AbstractScenario
    ) -> Tuple[FeaturesType, TargetsType]:
        computed_features = self._feature_preprocessor.compute_features(scenario)
        return computed_features[0], computed_features[1]

    @staticmethod
    def _strip_historical_feature(data: Dict[str, Any]) -> Dict[str, Any]:
        model_input_keys = {
            "agent",
            "map",
            "reference_line",
            "static_objects",
            "current_state",
            "origin",
            "angle",
        }
        return {key: value for key, value in data.items() if key in model_input_keys}


def create_dataset(
    samples: List[AbstractScenario],
    feature_preprocessor: FeaturePreprocessor,
    dataset_fraction: float,
    dataset_name: str,
    augmentors: Optional[List[AbstractAugmentor]] = None,
    use_lhpf_history: bool = False,
    lhpf_current_iteration: int = 1,
    lhpf_previous_iteration_delta: int = 1,
) -> torch.utils.data.Dataset:
    """
    Create a dataset from a list of samples.
    :param samples: List of dataset candidate samples.
    :param feature_preprocessor: Feature preprocessor object.
    :param dataset_fraction: Fraction of the dataset to load.
    :param dataset_name: Set name (train/val/test).
    :param scenario_type_loss_weights: Dictionary of scenario type loss weights.
    :param augmentors: List of augmentor objects for providing data augmentation to data samples.
    :return: The instantiated torch dataset.
    """
    # Sample the desired fraction from the total samples
    num_keep = max(1, int(len(samples) * dataset_fraction))
    num_keep = min(len(samples), num_keep)
    selected_scenarios = random.sample(samples, num_keep)

    logger.info(f"Number of samples in {dataset_name} set: {len(selected_scenarios)}")
    if use_lhpf_history:
        return LHPFScenarioDataset(
            scenarios=selected_scenarios,
            feature_preprocessor=feature_preprocessor,
            augmentors=augmentors,
            current_iteration=lhpf_current_iteration,
            previous_iteration_delta=lhpf_previous_iteration_delta,
        )

    return ScenarioDataset(
        scenarios=selected_scenarios,
        feature_preprocessor=feature_preprocessor,
        augmentors=augmentors,
    )


def distributed_weighted_sampler_init(
    scenario_dataset: ScenarioDataset,
    scenario_sampling_weights: Dict[str, float],
    replacement: bool = True,
) -> WeightedRandomSampler:
    """
    Initiliazes WeightedSampler object with sampling weights for each scenario_type and returns it.
    :param scenario_dataset: ScenarioDataset object
    :param replacement: Samples with replacement if True. By default set to True.
    return: Initialized Weighted sampler
    """
    scenarios = scenario_dataset._scenarios
    if (
        not replacement
    ):  # If we don't sample with replacement, then all sample weights must be nonzero
        assert all(
            w > 0 for w in scenario_sampling_weights.values()
        ), "All scenario sampling weights must be positive when sampling without replacement."

    default_scenario_sampling_weight = 1.0

    scenario_sampling_weights_per_idx = [
        scenario_sampling_weights[scenario.scenario_type]
        if scenario.scenario_type in scenario_sampling_weights
        else default_scenario_sampling_weight
        for scenario in scenarios
    ]

    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=scenario_sampling_weights_per_idx,
        num_samples=len(scenarios),
        replacement=replacement,
    )

    distributed_weighted_sampler = DistributedSamplerWrapper(weighted_sampler)
    return distributed_weighted_sampler


class CustomDataModule(pl.LightningDataModule):
    """
    Datamodule wrapping all preparation and dataset creation functionality.
    """

    def __init__(
        self,
        feature_preprocessor: FeaturePreprocessor,
        splitter: AbstractSplitter,
        all_scenarios: List[AbstractScenario],
        train_fraction: float,
        val_fraction: float,
        test_fraction: float,
        dataloader_params: Dict[str, Any],
        scenario_type_sampling_weights: DictConfig,
        worker: WorkerPool,
        augmentors: Optional[List[AbstractAugmentor]] = None,
        use_lhpf_history: bool = False,
        lhpf_current_iteration: int = 1,
        lhpf_previous_iteration_delta: int = 1,
    ) -> None:
        """
        Initialize the class.
        :param feature_preprocessor: Feature preprocessor object.
        :param splitter: Splitter object used to retrieve lists of samples to construct train/val/test sets.
        :param train_fraction: Fraction of training examples to load.
        :param val_fraction: Fraction of validation examples to load.
        :param test_fraction: Fraction of test examples to load.
        :param dataloader_params: Parameter dictionary passed to the dataloaders.
        :param augmentors: Augmentor object for providing data augmentation to data samples.
        """
        super().__init__()

        assert train_fraction > 0.0, "Train fraction has to be larger than 0!"
        assert val_fraction > 0.0, "Validation fraction has to be larger than 0!"
        assert test_fraction >= 0.0, "Test fraction has to be larger/equal than 0!"

        # Datasets
        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

        # Feature computation
        self._feature_preprocessor = feature_preprocessor

        # Data splitter train/test/val
        self._splitter = splitter

        # Fractions
        self._train_fraction = train_fraction
        self._val_fraction = val_fraction
        self._test_fraction = test_fraction

        # Data loader for train/val/test
        self._dataloader_params = dataloader_params

        # Extract all samples
        self._all_samples = all_scenarios
        assert len(self._all_samples) > 0, "No samples were passed to the datamodule"

        # Scenario sampling weights
        self._scenario_type_sampling_weights = scenario_type_sampling_weights

        # Augmentation setup
        self._augmentors = augmentors

        # Worker for multiprocessing to speed up initialization of datasets
        self._worker = worker

        # LHPF training samples need a previous-step feature to build latent memory.
        self._use_lhpf_history = use_lhpf_history
        self._lhpf_current_iteration = lhpf_current_iteration
        self._lhpf_previous_iteration_delta = lhpf_previous_iteration_delta

    @property
    def feature_and_targets_builder(self) -> FeaturePreprocessor:
        """Get feature and target builders."""
        return self._feature_preprocessor

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset for each target set depending on the training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        if stage is None:
            return

        if stage == "fit":
            # Training Dataset
            train_samples = self._splitter.get_train_samples(
                self._all_samples, self._worker
            )
            assert len(train_samples) > 0, "Splitter returned no training samples"

            self._train_set = create_dataset(
                train_samples,
                self._feature_preprocessor,
                self._train_fraction,
                "train",
                self._augmentors,
                self._use_lhpf_history,
                self._lhpf_current_iteration,
                self._lhpf_previous_iteration_delta,
            )

            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
                use_lhpf_history=self._use_lhpf_history,
                lhpf_current_iteration=self._lhpf_current_iteration,
                lhpf_previous_iteration_delta=self._lhpf_previous_iteration_delta,
            )
        elif stage == "validate":
            # Validation Dataset
            val_samples = self._splitter.get_val_samples(
                self._all_samples, self._worker
            )
            assert len(val_samples) > 0, "Splitter returned no validation samples"

            self._val_set = create_dataset(
                val_samples,
                self._feature_preprocessor,
                self._val_fraction,
                "validation",
                use_lhpf_history=self._use_lhpf_history,
                lhpf_current_iteration=self._lhpf_current_iteration,
                lhpf_previous_iteration_delta=self._lhpf_previous_iteration_delta,
            )
        elif stage == "test":
            # Testing Dataset
            test_samples = self._splitter.get_test_samples(
                self._all_samples, self._worker
            )
            assert len(test_samples) > 0, "Splitter returned no test samples"

            self._test_set = create_dataset(
                test_samples,
                self._feature_preprocessor,
                self._test_fraction,
                "test",
                use_lhpf_history=self._use_lhpf_history,
                lhpf_current_iteration=self._lhpf_current_iteration,
                lhpf_previous_iteration_delta=self._lhpf_previous_iteration_delta,
            )
        else:
            raise ValueError(
                f'Stage must be one of ["fit", "validate", "test"], got ${stage}.'
            )

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after a training stage.
        This is called by every process in distributed training.
        :param stage: Stage of training, can be "fit" or "test".
        """
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the training dataloader.
        :raises RuntimeError: If this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._train_set is None:
            raise DataModuleNotSetupError

        # Initialize weighted sampler
        if self._scenario_type_sampling_weights.enable:
            weighted_sampler = distributed_weighted_sampler_init(
                scenario_dataset=self._train_set,
                scenario_sampling_weights=self._scenario_type_sampling_weights.scenario_type_weights,
            )
        else:
            weighted_sampler = None

        return torch.utils.data.DataLoader(
            dataset=self._train_set,
            shuffle=weighted_sampler is None,
            collate_fn=FeatureCollate(),
            sampler=weighted_sampler,
            **self._dataloader_params,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the validation dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._val_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._val_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Create the test dataloader.
        :raises RuntimeError: if this method is called without calling "setup()" first.
        :return: The instantiated torch dataloader.
        """
        if self._test_set is None:
            raise DataModuleNotSetupError

        return torch.utils.data.DataLoader(
            dataset=self._test_set,
            **self._dataloader_params,
            collate_fn=FeatureCollate(),
        )

    # ! Modified to adapt to newer version of pytorch-lightning
    def transfer_batch_to_device(
        self, batch: Tuple[FeaturesType, ...], device: torch.device, dataloader_idx: int
    ) -> Tuple[FeaturesType, ...]:
        """
        Transfer a batch to device.
        :param batch: Batch on origin device.
        :param device: Desired device.
        :return: Batch in new device.
        """
        return tuple(
            (
                move_features_type_to_device(batch[0], device),
                move_features_type_to_device(batch[1], device),
                batch[2],
            )
        )
