# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HEC-HMS Model Optimizer.

HEC-HMS-specific optimizer inheriting from BaseModelOptimizer.
Supports DDS and other iterative optimization algorithms.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.evaluation.metrics import calculate_all_metrics
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer


class HecHmsModelOptimizer(BaseModelOptimizer):
    """
    HEC-HMS-specific optimizer using the unified BaseModelOptimizer framework.

    Supports:
    - Standard iterative optimization (DDS, PSO, SCE-UA, DE)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        self.hechms_setup_dir = self.project_dir / 'settings' / 'HECHMS'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("HecHmsModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'HECHMS'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to HEC-HMS configuration (placeholder for in-memory model)."""
        return self.hechms_setup_dir / 'hechms_config.txt'

    def _create_parameter_manager(self):
        """Create HEC-HMS parameter manager."""
        from jhechms.calibration.parameter_manager import HecHmsParameterManager
        return HecHmsParameterManager(
            self.config,
            self.logger,
            self.hechms_setup_dir
        )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run HEC-HMS for final evaluation using best parameters."""
        best_result = self.get_best_result()
        best_params = best_result.get('params')

        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False

        self.worker.apply_parameters(best_params, self.hechms_setup_dir)

        return self.worker.run_model(
            self.config,
            self.hechms_setup_dir,
            output_dir,
            save_output=True
        )

    def run_final_evaluation(self, best_params: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Run final evaluation with consistent warmup handling for HEC-HMS."""
        self.logger.info("=" * 60)
        self.logger.info("RUNNING FINAL EVALUATION")
        self.logger.info("=" * 60)

        try:
            if not self.worker._initialized:
                if not self.worker.initialize():
                    self.logger.error("Failed to initialize HEC-HMS worker for final evaluation")
                    return None

            if not self.worker.apply_parameters(best_params, self.hechms_setup_dir):
                self.logger.error("Failed to apply best parameters for final evaluation")
                return None

            final_output_dir = self.results_dir / 'final_evaluation'
            final_output_dir.mkdir(parents=True, exist_ok=True)

            runoff = self.worker._run_simulation(
                self.worker._forcing,
                best_params
            )

            self.worker.save_output_files(
                runoff[self.worker.warmup_days:],
                final_output_dir,
                self.worker._time_index[self.worker.warmup_days:] if self.worker._time_index is not None else None
            )

            # Calculate metrics using worker's in-memory data
            obs = self.worker._observations
            time_index = getattr(self.worker, '_time_index', None)

            if obs is not None and len(obs) == len(runoff) and time_index is not None:
                calib_period = self._parse_period_config(
                    'calibration_period', 'CALIBRATION_PERIOD')
                eval_period = self._parse_period_config(
                    'evaluation_period', 'EVALUATION_PERIOD')

                # Warmup is skipped from the FULL simulation before filtering to
                # the calibration period, matching how each calibration
                # iteration was scored. The evaluation period starts after
                # warmup, so it needs no further skip.
                calib_metrics = self._calculate_period_metrics_inmemory(
                    runoff, obs, time_index, calib_period, 'Calib', skip_warmup=True)
                eval_metrics = {}
                if eval_period[0] and eval_period[1]:
                    eval_metrics = self._calculate_period_metrics_inmemory(
                        runoff, obs, time_index, eval_period, 'Eval', skip_warmup=False)

                if calib_metrics or eval_metrics:
                    all_metrics = {**calib_metrics, **eval_metrics}
                    # Unprefixed aliases keep older readers of final_metrics working.
                    for key, value in calib_metrics.items():
                        unprefixed = key.replace('Calib_', '')
                        all_metrics.setdefault(unprefixed, value)

                    kge_score = calib_metrics.get('Calib_KGE')
                    if kge_score is not None:
                        self.logger.info(
                            f"Final evaluation KGE: {kge_score:.4f}, "
                            f"NSE: {calib_metrics.get('Calib_NSE', float('nan')):.4f}")
                    self._log_final_evaluation_results(calib_metrics, eval_metrics)

                    return {
                        'final_metrics': all_metrics,
                        'calibration_metrics': calib_metrics,
                        'evaluation_metrics': eval_metrics,
                        'success': True,
                        'best_params': best_params,
                        'output_dir': str(final_output_dir),
                    }

            self.logger.warning("Could not calculate final metrics")
            return {'best_params': best_params, 'output_dir': str(final_output_dir)}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Final evaluation failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_period_config(self, attr_name: str, dict_key: str):
        """Parse a period configuration string into (start, end) timestamps."""
        period_str = self._get_config_value(
            lambda: getattr(self.config.domain, attr_name, ''),
            default='',
            dict_key=dict_key
        )
        if not period_str:
            return (None, None)
        try:
            dates = [d.strip() for d in str(period_str).split(',')]
            if len(dates) >= 2:
                return (pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
        except (ValueError, AttributeError) as e:
            self.logger.debug(f"Could not parse period string '{period_str}': {e}")
        return (None, None)

    def _calculate_period_metrics_inmemory(
        self,
        runoff: np.ndarray,
        observations: np.ndarray,
        time_index: 'pd.DatetimeIndex',
        period: tuple,
        prefix: str,
        skip_warmup: bool = True
    ) -> Dict[str, float]:
        """Calculate prefixed metrics for one period from in-memory arrays.

        Warmup is dropped from the FULL simulation before filtering to the
        period, which is how each calibration iteration was scored:
        run 2002-2009 -> drop 365 warmup days -> filter to 2004-2007. The
        evaluation period already starts after warmup, so it passes
        ``skip_warmup=False``.
        """
        try:
            warmup = getattr(self.worker, 'warmup_days', 0) or 0
            if skip_warmup and len(runoff) > warmup:
                runoff = runoff[warmup:]
                observations = observations[warmup:]
                time_index = time_index[warmup:]

            sim_series = pd.Series(runoff, index=time_index)
            obs_series = pd.Series(observations, index=time_index)

            if period[0] and period[1]:
                mask = (time_index >= period[0]) & (time_index <= period[1])
                sim_period, obs_period = sim_series[mask], obs_series[mask]
                self.logger.debug(
                    f"{prefix} period: {period[0]} to {period[1]}, {len(sim_period)} points")
            else:
                sim_period, obs_period = sim_series, obs_series

            common_idx = sim_period.index.intersection(obs_period.index)
            if len(common_idx) == 0:
                self.logger.warning(f"No common indices for {prefix} period")
                return {}

            sim_aligned = sim_period.loc[common_idx].values
            obs_aligned = obs_period.loc[common_idx].values
            valid = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
            sim_valid, obs_valid = sim_aligned[valid], obs_aligned[valid]

            if len(sim_valid) < 10:
                self.logger.warning(
                    f"Insufficient valid points for {prefix} metrics: {len(sim_valid)}")
                return {}

            metrics_result = calculate_all_metrics(
                pd.Series(obs_valid), pd.Series(sim_valid))
            return {f"{prefix}_{k}": v for k, v in metrics_result.items()}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Failed to calculate {prefix} metrics: {e}")
            return {}
