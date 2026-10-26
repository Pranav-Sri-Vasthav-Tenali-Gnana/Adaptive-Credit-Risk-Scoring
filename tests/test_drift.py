"""
test_drift.py
-------------
Unit tests for PSI and KS test drift detection modules.
Run with: pytest tests/test_drift.py -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift.psi import calculate_psi_for_feature, calculate_psi_all_features
from src.drift.ks_test import ks_test_labels, ks_test_features


class TestPSI:

    def test_identical_distributions_low_psi(self):
        """Identical distributions should give PSI close to 0."""
        np.random.seed(42)
        data = np.random.normal(10000, 2000, 1000)
        psi = calculate_psi_for_feature(data, data)
        assert psi < 0.05, f"PSI for identical distributions should be near 0, got {psi}"

    def test_shifted_distribution_high_psi(self):
        """Significantly shifted distribution should give PSI > 0.2."""
        np.random.seed(42)
        ref = np.random.normal(10000, 1000, 1000)
        cur = np.random.normal(20000, 1000, 1000)  # Very different mean
        psi = calculate_psi_for_feature(ref, cur)
        assert psi > 0.2, f"PSI for highly shifted distribution should be > 0.2, got {psi}"

    def test_psi_all_features_returns_dict(self):
        """calculate_psi_all_features should return a dict with correct keys."""
        np.random.seed(42)
        ref_df = pd.DataFrame({
            "loan_amnt": np.random.normal(10000, 2000, 500),
            "dti": np.random.normal(15, 5, 500)
        })
        cur_df = pd.DataFrame({
            "loan_amnt": np.random.normal(12000, 2000, 500),
            "dti": np.random.normal(15.1, 5, 500)
        })
        result = calculate_psi_all_features(ref_df, cur_df, ["loan_amnt", "dti"])
        assert "loan_amnt" in result
        assert "dti" in result
        assert isinstance(result["loan_amnt"], float)

    def test_psi_handles_missing_feature(self):
        """Should not crash if feature is missing from one DataFrame."""
        ref_df = pd.DataFrame({"loan_amnt": np.random.normal(10000, 2000, 500)})
        cur_df = pd.DataFrame({"dti": np.random.normal(15, 5, 500)})
        result = calculate_psi_all_features(ref_df, cur_df, ["loan_amnt"])
        assert result == {}


class TestKSTest:

    def test_same_distribution_no_drift(self):
        """Same distribution should not trigger drift (p > 0.05)."""
        np.random.seed(42)
        labels = np.random.binomial(1, 0.15, 1000)
        result = ks_test_labels(labels, labels)
        assert not result["drift_detected"]
        assert result["ks_statistic"] == 0.0

    def test_different_distribution_drift_detected(self):
        """Very different distributions should trigger drift."""
        np.random.seed(42)
        ref = np.random.binomial(1, 0.10, 1000)
        cur = np.random.binomial(1, 0.40, 1000)
        result = ks_test_labels(ref, cur)
        assert result["drift_detected"], "Drift should be detected for very different distributions"

    def test_result_has_expected_keys(self):
        """Result dict should have all expected keys."""
        np.random.seed(42)
        ref = np.random.binomial(1, 0.15, 500)
        cur = np.random.binomial(1, 0.20, 500)
        result = ks_test_labels(ref, cur)
        expected_keys = [
            "ks_statistic", "p_value", "drift_detected",
            "reference_default_rate", "current_default_rate"
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_default_rates_are_correct(self):
        """Default rates should match input data."""
        ref = np.array([0, 0, 0, 1, 1])  # 40% default
        cur = np.array([0, 1, 1, 1, 1])  # 80% default
        result = ks_test_labels(ref, cur)
        assert abs(result["reference_default_rate"] - 0.4) < 0.01
        assert abs(result["current_default_rate"] - 0.8) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
