"""
Root cause analysis evaluator.

This module evaluates root cause localization results against ground truth,
following the RQ2 benchmark implementation for RCA evaluation.

Metrics:
- acc@k: Accuracy at k = |top_k ∩ ground_truth| / min(k, |ground_truth|)
- avg@k: Average accuracy - average of acc@1, acc@2, ..., acc@k
- P (Precision): Mean of per-case precisions (correct / total_predictions for each case)
- R (Recall): Dataset-level recall - #cases_with_correct_root_cause / total_cases
"""

from typing import List, Set, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RCAEvaluator:
    """
    Evaluator for root cause analysis results.

    Metrics:
    - acc@k: Accuracy at k = |top_k ∩ ground_truth| / min(k, |ground_truth|)
    - avg@k: Average of acc@1 to acc@k
    - P: Mean of per-case precisions = mean(correct_i / predictions_i)
    - R: Dataset-level recall = #cases_with_correct_root_cause / total_cases
    """

    def __init__(self):
        """Initialize the RCA evaluator."""
        # Track case-level results for dataset-level metrics
        self.case_results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self,
                 predicted_root_causes: List[str],
                 ground_truth_root_causes: List[str],
                 return_top_k: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single RCA result.

        Args:
            predicted_root_causes: Ranked list of predicted root causes
            ground_truth_root_causes: List of ground truth root causes (from root_cause.txt first item)
            return_top_k: Whether to compute Top-K accuracy

        Returns:
            Dictionary with acc@k, avg@k, P, R metrics for this case
        """
        predicted_set = set(predicted_root_causes)
        gt_set = set(ground_truth_root_causes)
        predicted_count = len(predicted_root_causes)

        # Compute acc@k for k = 1, 2, 3, 4, 5
        # acc@k = |top_k ∩ ground_truth| / min(k, |ground_truth|)
        # When k > predicted_count, top_k uses all available predictions
        acc_results = {}
        gt_count = len(gt_set)

        for k in [1, 2, 3, 4, 5]:
            # Get top_k predictions (min handles k > predicted_count case)
            top_k = set(predicted_root_causes[:min(k, predicted_count)])
            denominator = min(k, gt_count)

            if denominator > 0 and gt_set:
                acc_results[f'acc@{k}'] = len(top_k & gt_set) / denominator
            else:
                acc_results[f'acc@{k}'] = 0.0

        # Compute avg@k = average of acc@1...acc@k
        avg_results = {}
        for k in [1, 2, 3, 4, 5]:
            acc_values = [acc_results[f'acc@{j}'] for j in range(1, k + 1)]
            avg_results[f'avg@{k}'] = sum(acc_values) / k

        # Compute Precision (P) = correct / total predictions
        if predicted_root_causes:
            precision = len(predicted_set & gt_set) / len(predicted_root_causes)
        else:
            precision = 0.0

        # Compute Recall (R) = 1 if gt in predictions else 0 (for this case)
        recall = 1.0 if (predicted_set & gt_set) else 0.0

        results = {
            **acc_results,
            **avg_results,
            'precision': precision,
            'recall': recall,
            'predicted_count': len(predicted_root_causes),
            'ground_truth_count': len(ground_truth_root_causes),
            # Include raw counts for dataset-level aggregation
            'correct_count': len(predicted_set & gt_set),
            'total_predictions': len(predicted_root_causes)
        }

        # Store case result for dataset-level aggregation
        # Keep full ranked list for acc@k computation at dataset level
        self.case_results.append({
            'predicted_list': predicted_root_causes,  # Full ranked list
            'predicted_set': predicted_set,
            'ground_truth': gt_set,
            'predicted_count': len(predicted_root_causes),
            'gt_count': len(ground_truth_root_causes),
            'correct': len(predicted_set & gt_set)
        })

        self.logger.info(
            f"RCA Evaluation: avg@5={avg_results['avg@5']:.3f}, "
            f"acc@1={acc_results['acc@1']:.3f}, acc@3={acc_results['acc@3']:.3f}, "
            f"P={precision:.3f}, R={recall:.3f}"
        )

        return results
