"""Validation analyzer that reproduces verl's `val-core/...` metric family.

This exists so SPO runs log the *exact same* wandb keys as grpocredit (which uses
verl directly), enabling an apple-to-apple overlay of the two runs. The metric
math (per-prompt mean / std, bootstrapped best/worst@k) and the wandb key strings
are ported verbatim from the vendored verl used by grpocredit:

    external/verl/verl/trainer/ppo/metric_utils.py  (bootstrap_metric, process_validation_metrics)
    external/verl/verl/trainer/ppo/ray_trainer.py   (val-core/val-aux key construction)

The per-sample binary correctness that feeds the aggregation is produced with
SPO's own grading (`Task.extract_predicted_answer_from_text` + `Task.grade_answer`),
identical to `MATHTask.evaluate_predictions`, so reward==1.0 iff that sample is
graded correct.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from treetune import logging_utils
from treetune.analyzers import Analyzer

logger = logging_utils.get_logger(__name__)


def _bootstrap_metric(
    data: List[Any],
    subset_size: int,
    reduce_fns: List[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[Tuple[float, float]]:
    """Verbatim port of verl `bootstrap_metric` (metric_utils.py:247-285)."""
    np.random.seed(seed)
    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


@Analyzer.register("mean_at_k")
class MeanAtKAnalyzer(Analyzer):
    """Logs verl-style `val-core/{data_source}/{var_name}/mean@k` validation metrics.

    Only deepseekmath-on-math_v2 is supported: every other model (qwen, R1-Distill)
    uses different grpocredit sampling/reward settings that are not ported here, so
    running this analyzer for them would produce a misleading comparison. Those cases
    raise NotImplementedError instead.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        data_source: str = "math_v2",
        var_name: str = "reward",
        n_bootstrap: int = 1000,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name_or_path = model_name_or_path
        self.data_source = data_source
        self.var_name = var_name
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self._assert_supported_setup()

        self.result_dir = self.runtime._get_result_dir()
        assert self.result_dir is not None, "Result directory is not set."
        if not self.result_dir.exists():
            raise ValueError(f"Result directory {self.result_dir} does not exist.")

    def _assert_supported_setup(self) -> None:
        name = (self.model_name_or_path or "").lower()
        unsupported = [m for m in ("qwen", "distill", "deepseek-r1") if m in name]
        if unsupported:
            raise NotImplementedError(
                f"MeanAtKAnalyzer only supports deepseekmath on math_v2 with "
                f"grpocredit-matched settings. Model '{self.model_name_or_path}' "
                f"matches {unsupported}; its grpocredit sampling/reward settings are "
                f"not ported, so an apple-to-apple comparison is not valid here."
            )
        if "deepseekmath" not in name:
            raise NotImplementedError(
                f"MeanAtKAnalyzer only supports deepseekmath models; got "
                f"'{self.model_name_or_path}'."
            )
        if self.data_source != "math_v2":
            raise NotImplementedError(
                f"MeanAtKAnalyzer only supports data_source='math_v2'; got "
                f"'{self.data_source}'."
            )

    def get_analysis_id(self) -> str:
        return super().get_analysis_id() + self.result_dir.name

    def _grade_per_sample_rewards(
        self, predictions: List[List[str]], references: Dataset
    ) -> List[List[float]]:
        """Per-problem list of per-sample 0/1 rewards (verl's `var_vals`).

        Mirrors MATHTask.evaluate_predictions so a sample's reward is 1.0 iff SPO
        grades it correct.
        """
        assert len(predictions) == len(references)
        assert len(predictions) > 0, "No predictions provided."

        per_problem_rewards: List[List[float]] = []
        for solution_candidates, ref in zip(predictions, references):
            gold_answer = ref["answer"]
            problem = ref["problem"]
            assert len(solution_candidates) > 0

            answer_candidates = [
                self.task.extract_predicted_answer_from_text(sol, problem=problem)
                for sol in solution_candidates
            ]
            grading_results = [
                self.task.grade_answer(
                    given_answer=ans, ground_truth=gold_answer, item=ref
                )
                for ans in answer_candidates
            ]
            per_problem_rewards.append([float(bool(g)) for g in grading_results])

        return per_problem_rewards

    def _metrics_for_one_problem(self, var_vals: List[float]) -> Dict[str, float]:
        """Port of the per-prompt block of verl `process_validation_metrics`."""
        metric: Dict[str, float] = {}
        n_resps = len(var_vals)
        metric[f"mean@{n_resps}"] = float(np.mean(var_vals))

        if n_resps > 1:
            metric[f"std@{n_resps}"] = float(np.std(var_vals))

            ns = []
            n = 2
            while n < n_resps:
                ns.append(n)
                n *= 2
            ns.append(n_resps)

            for n in ns:
                [(bon_mean, bon_std), (won_mean, won_std)] = _bootstrap_metric(
                    data=var_vals,
                    subset_size=n,
                    reduce_fns=[np.max, np.min],
                    n_bootstrap=self.n_bootstrap,
                    seed=self.seed,
                )
                metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = (
                    float(bon_mean),
                    float(bon_std),
                )
                metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = (
                    float(won_mean),
                    float(won_std),
                )

        return metric

    def _build_metric_dict(self, metric2val: Dict[str, float]) -> Dict[str, float]:
        """Port of verl ray_trainer val-core/val-aux key construction.

        We only ever have one variable ("reward"), so `core_var == var_name` and the
        resulting keys are byte-identical to grpocredit's, e.g.
        `val-core/math_v2/reward/mean@16`.
        """
        core_var = self.var_name
        n_max = max(
            int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()
        )

        metric_dict: Dict[str, float] = {}
        for metric_name, metric_val in metric2val.items():
            if (
                self.var_name == core_var
                and any(metric_name.startswith(p) for p in ["mean", "maj", "best"])
                and (f"@{n_max}" in metric_name)
            ):
                metric_sec = "val-core"
            else:
                metric_sec = "val-aux"
            key = f"{metric_sec}/{self.data_source}/{self.var_name}/{metric_name}"
            metric_dict[key] = metric_val
        return metric_dict

    def analyze(self) -> Dict[str, float]:
        super().analyze()

        logger.info(f"Analyzing {self.result_dir} for mean@k metrics...")
        output_dataset = Dataset.load_from_disk(str(self.result_dir))

        assert (
            "_treetune__candidate_answers" in output_dataset.features
        ), "The dataset does not contain the candidate answers."

        predictions = output_dataset["_treetune__candidate_answers"]
        references = output_dataset

        per_problem_rewards = self._grade_per_sample_rewards(predictions, references)

        n_resps_counts = sorted({len(r) for r in per_problem_rewards})
        if len(n_resps_counts) != 1:
            # verl assumes a uniform sample count per prompt; ragged counts split
            # across different mean@k keys and would break the comparison.
            logger.warning(
                f"Non-uniform sample counts per problem: {n_resps_counts}. "
                f"verl assumes a fixed n; mean@k keys may be split."
            )
        logger.info(
            f"Computing mean@k over {len(per_problem_rewards)} problems "
            f"(samples per problem: {n_resps_counts})."
        )

        # Aggregate per-prompt metrics across prompts (verl averages by metric name).
        per_metric_vals: Dict[str, List[float]] = defaultdict(list)
        for var_vals in per_problem_rewards:
            for metric_name, value in self._metrics_for_one_problem(var_vals).items():
                per_metric_vals[metric_name].append(value)
        metric2val = {
            metric_name: float(np.mean(vals))
            for metric_name, vals in per_metric_vals.items()
        }

        metric_dict = self._build_metric_dict(metric2val)

        # Persist locally (log.json) and log the exact keys to wandb. We bypass
        # self.log_metrics on purpose: it force-prepends self.plot_prefix, which
        # would change the key strings and defeat the apple-to-apple comparison.
        self.log(metric_dict)
        if self.cloud_logger is not None:
            payload = dict(metric_dict)
            if self.iteration is not None:
                payload["train/iteration"] = self.iteration
            if self.global_step is not None:
                payload["train/global_step"] = self.global_step
            self.cloud_logger.log(payload)

        logger.info(f"mean@k metrics: {metric_dict}")
        return metric_dict
