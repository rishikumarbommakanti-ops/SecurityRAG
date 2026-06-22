"""Tests for the enterprise AI dashboard."""

from datetime import datetime

from app.dashboard.dashboard import DashboardMetrics, EnterpriseAIDashboard


def test_dashboard_metrics_initialization():
    """Test DashboardMetrics initialization."""
    metrics = DashboardMetrics()

    assert metrics.total_documents == 0
    assert metrics.total_queries == 0
    assert metrics.local_answers == 0
    assert metrics.gemini_answers == 0
    assert metrics.average_confidence == 0.0
    assert metrics.documents_by_source == {}
    assert metrics.duplicate_count == 0
    assert metrics.coverage_percentage == 0.0


def test_dashboard_metrics_to_dict():
    """Test DashboardMetrics export to dictionary."""
    metrics = DashboardMetrics()
    metrics.total_documents = 100
    metrics.local_answers = 20
    metrics.gemini_answers = 30

    metrics_dict = metrics.to_dict()

    assert metrics_dict["total_documents"] == 100
    assert metrics_dict["local_answers"] == 20
    assert metrics_dict["gemini_answers"] == 30


def test_enterprise_dashboard_initialization():
    """Test EnterpriseAIDashboard initialization."""
    dashboard = EnterpriseAIDashboard()

    assert dashboard.metrics is not None
    assert dashboard.metrics.total_documents == 0


def test_enterprise_dashboard_update_metrics():
    """Test updating dashboard metrics."""
    dashboard = EnterpriseAIDashboard()

    test_metrics = {
        "total_documents": 500,
        "total_queries": 50,
        "local_answers": 30,
        "gemini_answers": 20,
        "average_confidence": 0.85,
        "documents_by_source": {"MITRE ATT&CK": 200, "Sigma": 300},
        "duplicate_count": 5,
        "coverage_percentage": 75.5,
    }

    dashboard.update_metrics(test_metrics)

    assert dashboard.metrics.total_documents == 500
    assert dashboard.metrics.total_queries == 50
    assert dashboard.metrics.local_answers == 30
    assert dashboard.metrics.gemini_answers == 20
    assert dashboard.metrics.average_confidence == 0.85
    assert dashboard.metrics.documents_by_source == {"MITRE ATT&CK": 200, "Sigma": 300}
    assert dashboard.metrics.duplicate_count == 5
    assert dashboard.metrics.coverage_percentage == 75.5


def test_dashboard_metrics_with_charts_data():
    """Test dashboard with chart data."""
    dashboard = EnterpriseAIDashboard()

    test_metrics = {
        "total_documents": 1000,
        "total_queries": 100,
        "local_answers": 60,
        "gemini_answers": 40,
        "confidence_scores": [0.9, 0.85, 0.92, 0.78, 0.88],
        "response_times": [0.5, 0.6, 0.4, 0.7, 0.5],
        "mitre_techniques": {"T1059": 45, "T1086": 38},
        "sigma_categories": {"Process Execution": 150, "Network Connection": 120},
        "knowledge_categories": {"Detection": 450, "Forensics": 240},
    }

    dashboard.update_metrics(test_metrics)

    assert len(dashboard.metrics.confidence_scores) == 5
    assert len(dashboard.metrics.response_times) == 5
    assert len(dashboard.metrics.mitre_techniques) == 2
    assert len(dashboard.metrics.sigma_categories) == 2
    assert len(dashboard.metrics.knowledge_categories) == 2
