# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""CPU regressions for the V0 trainers' tracking lifecycle."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import Role

from verl_omni.trainer.diffusion import ray_diffusion_trainer


@pytest.fixture(
    params=[ray_diffusion_trainer.PolicyGradientRayTrainer, ray_diffusion_trainer.DirectPreferenceRayTrainer],
    ids=["policy_gradient", "direct_preference"],
)
def trainer_context(request, monkeypatch):
    """Keep real trainer construction while replacing unrelated data and skip setup."""
    direct_preference = request.param is ray_diffusion_trainer.DirectPreferenceRayTrainer
    config = OmegaConf.create(
        {
            "algorithm": {
                "trainer_type": "direct_preference" if direct_preference else "policy_gradient",
                "sample_source": "online",
            },
            "actor_rollout_ref": {
                "hybrid_engine": True,
                "model": {"lora_rank": 0},
                "actor": {
                    "use_kl_loss": False,
                    "use_distill_loss": False,
                    "diffusion_loss": {"loss_mode": "dpo" if direct_preference else "flow_grpo"},
                    "optim": {"total_training_steps": None},
                },
            },
            "reward": {"reward_model": {"enable": False}},
            "trainer": {
                "device": "cpu",
                "project_name": "logger-cpu",
                "experiment_name": "${trainer.project_name}-run",
                "logger": ["wandb"],
                "val_before_train": True,
                "val_only": True,
            },
        }
    )
    events = []

    def create_dataloader(trainer, train_dataset, val_dataset, collate_fn, train_sampler):
        """Supply CPU batches and the optimizer update normally made by data setup."""
        events.append("data")
        trainer.train_dataloader = [None]
        trainer.total_training_steps = 3
        trainer.config.actor_rollout_ref.actor.optim.total_training_steps = trainer.total_training_steps

    def initialize_skip_manager(config):
        """Record skip setup without modifying process-wide SkipManager state."""
        events.append("skip")

    monkeypatch.setattr(ray_diffusion_trainer.BaseRayDiffusionTrainer, "_create_dataloader", create_dataloader)
    monkeypatch.setattr(ray_diffusion_trainer, "init_skip_manager", initialize_skip_manager)
    return SimpleNamespace(trainer_class=request.param, config=config, events=events, resource_pool_manager=Mock())


def test_tracking_failure_precedes_worker_initialization(monkeypatch, trainer_context):
    """A failed W&B initialization aborts construction before any worker or pool setup."""
    context = trainer_context
    tracking = Mock(side_effect=RuntimeError("tracking initialization failed"))
    init_workers = Mock()
    colocated_workers = Mock()
    online_stack = Mock()
    monkeypatch.setattr(ray_diffusion_trainer, "Tracking", tracking)
    monkeypatch.setattr(context.trainer_class, "init_workers", init_workers)
    monkeypatch.setattr(context.trainer_class, "_init_colocated_workers", colocated_workers)
    monkeypatch.setattr(context.trainer_class, "_init_online_rollout_stack", online_stack)

    with pytest.raises(RuntimeError, match="tracking initialization failed"):
        trainer = context.trainer_class(
            config=context.config,
            tokenizer=None,
            role_worker_mapping={Role.ActorRollout: Mock()},
            resource_pool_manager=context.resource_pool_manager,
        )
        trainer.init_workers()
        trainer.fit()

    assert context.events == ["data", "skip"]
    tracking.assert_called_once()
    init_workers.assert_not_called()
    colocated_workers.assert_not_called()
    online_stack.assert_not_called()
    context.resource_pool_manager.create_resource_pool.assert_not_called()


def test_fit_reuses_constructor_logger(monkeypatch, trainer_context):
    """Both real V0 fit loops log through the one run created after CPU data setup."""
    context = trainer_context
    logger = Mock()

    def initialize_tracking(**kwargs):
        """Record creation so its ordering can be checked against data and workers."""
        context.events.append("tracking")
        return logger

    def initialize_colocated_workers():
        """Record worker initialization without creating Ray actors or accelerator pools."""
        context.events.append("workers")
        return "actor_pool"

    tracking = Mock(side_effect=initialize_tracking)
    monkeypatch.setattr(ray_diffusion_trainer, "Tracking", tracking)
    monkeypatch.setattr(ray_diffusion_trainer, "invoke_experiment", Mock())
    trainer = context.trainer_class(
        config=context.config,
        tokenizer=None,
        role_worker_mapping={Role.ActorRollout: Mock()},
        resource_pool_manager=context.resource_pool_manager,
    )
    assert trainer.logger is logger
    assert context.events == ["data", "skip", "tracking"]
    tracking.assert_called_once_with(
        project_name="logger-cpu",
        experiment_name="logger-cpu-run",
        default_backend=context.config.trainer.logger,
        config=OmegaConf.to_container(context.config, resolve=True),
    )
    recorded_config = tracking.call_args.kwargs["config"]
    assert recorded_config["actor_rollout_ref"]["actor"]["optim"]["total_training_steps"] == 3
    assert recorded_config["trainer"]["experiment_name"] == "logger-cpu-run"

    online_stack = Mock()
    monkeypatch.setattr(trainer, "_init_colocated_workers", initialize_colocated_workers)
    monkeypatch.setattr(trainer, "_init_online_rollout_stack", online_stack)
    trainer.init_workers()
    online_stack.assert_called_once_with("actor_pool")
    assert context.events == ["data", "skip", "tracking", "workers"]

    trainer.checkpoint_manager = Mock()
    monkeypatch.setattr(trainer, "_load_checkpoint", Mock())
    validation = Mock(return_value={"val/reward": 1.0})
    monkeypatch.setattr(trainer, "_validate", validation)
    for expected_logs in (1, 2):
        trainer.fit()
        assert trainer.logger is logger
        tracking.assert_called_once()
        assert logger.log.call_count == expected_logs
        logger.log.assert_called_with(data={"val/reward": 1.0}, step=0)
    assert validation.call_count == 2
