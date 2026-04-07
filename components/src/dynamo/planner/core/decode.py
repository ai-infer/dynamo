# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Optional

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.core.base import BasePlanner
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DecodePlanner(BasePlanner):
    component_type = SubComponentType.DECODE

    def load_plan_adjustment(self) -> Optional[int]:
        """Load-based scaling decision for decode using FPM data.

        For each engine, estimates next decode ITL:
        - Uses scheduled + queued decode KV tokens + avg decode length
        - Predicts wall time via regression

        Scale up if ALL engines' estimated ITL > SLA.
        Scale down if ALL engines' estimated ITL < SLA * sensitivity.
        """
        if not self.itl_regression.has_sufficient_data():
            logger.info(
                f"ITL regression: insufficient data ({self.itl_regression.num_observations}"
                f"/{self.itl_regression.min_observations}), skipping load-based scaling"
            )
            return None

        fpm_stats = self._get_fpm_stats()
        if not fpm_stats:
            return None

        num_workers = self.shared_state.num_d_workers
        if num_workers == 0:
            return None

        estimated_itls: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            scheduled_kv = fpm.scheduled_requests.sum_decode_kv_tokens
            queued_kv = fpm.queued_requests.sum_decode_kv_tokens
            est = self.itl_regression.estimate_next_itl(
                scheduled_decode_kv=scheduled_kv,
                queued_decode_kv=queued_kv,
            )
            if est is None:
                continue
            est_ms = est * 1000
            estimated_itls.append(est_ms)
            logger.info(
                f"Decode engine {wid}:dp{dp}: estimated ITL {est_ms:.2f}ms "
                f"(sched_kv={scheduled_kv}, queued_kv={queued_kv}, "
                f"avg_decode_len={self.itl_regression.avg_decode_length:.1f})"
            )

        return self._load_based_scaling_decision_from_estimates(
            estimates=estimated_itls,
            sla=self.config.itl,
            num_workers=num_workers,
            label="decode ITL",
        )

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        engine_thpt, _ = self.itl_regression.find_best_engine_decode_thpt(
            itl=self.config.itl, context_length=next_isl + next_osl / 2
        )
        if engine_thpt <= 0:
            logger.warning(
                f"engine_decode_thpt is {engine_thpt}, falling back to min_endpoint"
            )
            return self.config.min_endpoint
        pred_decode_throughput = (
            next_num_req * next_osl / self.config.throughput_adjustment_interval
        )
        next_num_d = math.ceil(pred_decode_throughput / engine_thpt)
        next_num_d = max(next_num_d, self.config.min_endpoint)
        logger.info(
            f"Decode calculation: {pred_decode_throughput:.2f}(demand tps) / "
            f"{engine_thpt:.2f}(engine_thpt) = {next_num_d}(num_d)"
        )
        return next_num_d

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_d.set(desired_replicas)
