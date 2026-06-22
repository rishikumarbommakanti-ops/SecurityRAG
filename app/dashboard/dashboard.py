"""Enterprise AI Security Dashboard for SecurityRAG.

Provides comprehensive analytics, metrics, and visualizations for the
hybrid RAG system including local answers, Gemini answers, confidence
scores, threat intelligence analysis, and knowledge health monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


class DashboardMetrics:
    """Aggregated metrics for dashboard display."""

    def __init__(self) -> None:
        self.total_documents = 0
        self.total_queries = 0
        self.local_answers = 0
        self.gemini_answers = 0
        self.average_confidence = 0.0
        self.documents_by_source = {}
        self.mitre_techniques = {}
        self.sigma_categories = {}
        self.knowledge_categories = {}
        self.last_update_time = None
        self.duplicate_count = 0
        self.coverage_percentage = 0.0
        self.recent_questions = []
        self.response_times = []
        self.confidence_scores = []

    def to_dict(self) -> dict:
        """Export metrics as dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_queries": self.total_queries,
            "local_answers": self.local_answers,
            "gemini_answers": self.gemini_answers,
            "average_confidence": self.average_confidence,
            "documents_by_source": self.documents_by_source,
            "mitre_techniques": self.mitre_techniques,
            "sigma_categories": self.sigma_categories,
            "knowledge_categories": self.knowledge_categories,
            "last_update_time": self.last_update_time,
            "duplicate_count": self.duplicate_count,
            "coverage_percentage": self.coverage_percentage,
            "recent_questions": self.recent_questions,
            "response_times": self.response_times,
            "confidence_scores": self.confidence_scores,
        }


class EnterpriseAIDashboard:
    """Enterprise-grade AI Security Operations dashboard."""

    DARK_THEME_CONFIG = {
        "plot_bgcolor": "#1a1a2e",
        "paper_bgcolor": "#0f0f1e",
        "font": {"color": "#e0e0e0"},
    }

    def __init__(self) -> None:
        self.metrics = DashboardMetrics()

    def update_metrics(self, metrics_dict: Optional[Dict[str, Any]] = None) -> None:
        """Update dashboard metrics from provided data."""
        if metrics_dict is None:
            metrics_dict = {}

        self.metrics.total_documents = metrics_dict.get("total_documents", 0)
        self.metrics.total_queries = metrics_dict.get("total_queries", 0)
        self.metrics.local_answers = metrics_dict.get("local_answers", 0)
        self.metrics.gemini_answers = metrics_dict.get("gemini_answers", 0)
        self.metrics.average_confidence = metrics_dict.get("average_confidence", 0.0)
        self.metrics.documents_by_source = metrics_dict.get("documents_by_source", {})
        self.metrics.mitre_techniques = metrics_dict.get("mitre_techniques", {})
        self.metrics.sigma_categories = metrics_dict.get("sigma_categories", {})
        self.metrics.knowledge_categories = metrics_dict.get("knowledge_categories", {})
        self.metrics.last_update_time = metrics_dict.get("last_update_time")
        self.metrics.duplicate_count = metrics_dict.get("duplicate_count", 0)
        self.metrics.coverage_percentage = metrics_dict.get("coverage_percentage", 0.0)
        self.metrics.recent_questions = metrics_dict.get("recent_questions", [])
        self.metrics.response_times = metrics_dict.get("response_times", [])
        self.metrics.confidence_scores = metrics_dict.get("confidence_scores", [])

    def render_executive_dashboard(self) -> None:
        """Render the executive dashboard with key metrics."""
        st.header("📊 Executive Dashboard")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label="Total Documents",
                value=self.metrics.total_documents,
                delta="Knowledge Base Size",
            )

        with col2:
            st.metric(
                label="Knowledge Coverage",
                value=f"{self.metrics.coverage_percentage:.1f}%",
                delta="Domain Completeness",
            )

        with col3:
            st.metric(
                label="Local Answers",
                value=self.metrics.local_answers,
                delta="Deterministic Responses",
            )

        with col4:
            st.metric(
                label="Gemini Answers",
                value=self.metrics.gemini_answers,
                delta="AI-Generated Responses",
            )

        with col5:
            st.metric(
                label="AI Confidence",
                value=f"{self.metrics.average_confidence:.2f}",
                delta="Avg Score (0-1)",
            )

    def render_ai_analytics(self) -> None:
        """Render AI analytics with interactive charts."""
        st.header("📈 AI Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Answer Source Distribution")
            if self.metrics.local_answers > 0 or self.metrics.gemini_answers > 0:
                source_data = {
                    "Source": ["Local", "Gemini"],
                    "Count": [self.metrics.local_answers, self.metrics.gemini_answers],
                }
                fig = px.pie(
                    source_data,
                    names="Source",
                    values="Count",
                    color_discrete_map={"Local": "#10b981", "Gemini": "#f59e0b"},
                )
                fig.update_layout(**self.DARK_THEME_CONFIG, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query data available yet")

        with col2:
            st.subheader("Confidence Distribution")
            if self.metrics.confidence_scores:
                fig = go.Figure(
                    data=[
                        go.Histogram(
                            x=self.metrics.confidence_scores,
                            nbinsx=20,
                            marker_color="#6366f1",
                        )
                    ]
                )
                fig.update_layout(
                    title="Confidence Score Histogram",
                    xaxis_title="Confidence Score",
                    yaxis_title="Frequency",
                    **self.DARK_THEME_CONFIG,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No confidence data available yet")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Response Time Trends")
            if self.metrics.response_times:
                fig = go.Figure(
                    data=[
                        go.Scatter(
                            y=self.metrics.response_times,
                            mode="lines",
                            line=dict(color="#8b5cf6", width=2),
                        )
                    ]
                )
                fig.update_layout(
                    title="Response Times Over Time",
                    yaxis_title="Time (seconds)",
                    xaxis_title="Query Index",
                    **self.DARK_THEME_CONFIG,
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No response time data available yet")

        with col4:
            st.subheader("Query Trends")
            if self.metrics.total_queries > 0:
                trend_data = {
                    "Metric": ["Local", "Gemini"],
                    "Answers": [self.metrics.local_answers, self.metrics.gemini_answers],
                }
                fig = px.bar(
                    trend_data,
                    x="Metric",
                    y="Answers",
                    color="Metric",
                    color_discrete_map={"Local": "#10b981", "Gemini": "#f59e0b"},
                )
                fig.update_layout(**self.DARK_THEME_CONFIG, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No query data available yet")

    def render_threat_intelligence(self) -> None:
        """Render threat intelligence analysis."""
        st.header("🎯 Threat Intelligence")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("MITRE ATT&CK Techniques")
            if self.metrics.mitre_techniques:
                mitre_data = {
                    "Technique": list(self.metrics.mitre_techniques.keys())[:10],
                    "Count": list(self.metrics.mitre_techniques.values())[:10],
                }
                fig = px.bar(
                    mitre_data,
                    x="Count",
                    y="Technique",
                    orientation="h",
                    color="Count",
                    color_continuous_scale="Viridis",
                )
                fig.update_layout(**self.DARK_THEME_CONFIG, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No MITRE technique data available")

        with col2:
            st.subheader("Sigma Rule Categories")
            if self.metrics.sigma_categories:
                sigma_data = {
                    "Category": list(self.metrics.sigma_categories.keys())[:8],
                    "Count": list(self.metrics.sigma_categories.values())[:8],
                }
                fig = px.pie(
                    sigma_data,
                    names="Category",
                    values="Count",
                )
                fig.update_layout(**self.DARK_THEME_CONFIG, height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Sigma category data available")

        st.subheader("Knowledge Categories")
        if self.metrics.knowledge_categories:
            category_data = {
                "Category": list(self.metrics.knowledge_categories.keys()),
                "Documents": list(self.metrics.knowledge_categories.values()),
            }
            fig = px.bar(
                category_data,
                x="Category",
                y="Documents",
                color="Documents",
                color_continuous_scale="Blues",
            )
            fig.update_layout(**self.DARK_THEME_CONFIG, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No knowledge category data available")

    def render_knowledge_health(self) -> None:
        """Render knowledge health and maintenance metrics."""
        st.header("🏥 Knowledge Health")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Total Sources",
                value=len(self.metrics.documents_by_source),
                delta="Active Connectors",
            )

        with col2:
            st.metric(
                label="Duplicate Documents",
                value=self.metrics.duplicate_count,
                delta="Deduplication Needed",
            )

        with col3:
            if self.metrics.last_update_time:
                time_diff = datetime.now() - self.metrics.last_update_time
                hours_ago = int(time_diff.total_seconds() / 3600)
                st.metric(
                    label="Last Updated",
                    value=f"{hours_ago}h ago",
                    delta="Last sync time",
                )
            else:
                st.metric(label="Last Updated", value="Never", delta="No updates yet")

        with col4:
            st.metric(
                label="Coverage %",
                value=f"{self.metrics.coverage_percentage:.1f}%",
                delta="Knowledge completeness",
            )

        st.subheader("Documents by Source")
        if self.metrics.documents_by_source:
            source_data = {
                "Source": list(self.metrics.documents_by_source.keys()),
                "Documents": list(self.metrics.documents_by_source.values()),
            }
            fig = px.bar(
                source_data,
                x="Source",
                y="Documents",
                color="Documents",
                color_continuous_scale="Greens",
            )
            fig.update_layout(**self.DARK_THEME_CONFIG, height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No document source data available")

    def render_recent_activity(self) -> None:
        """Render recent activity log."""
        st.header("📋 Recent Activity")

        if self.metrics.recent_questions:
            st.subheader("Recent Questions")
            for i, question in enumerate(self.metrics.recent_questions[-10:], 1):
                with st.container():
                    cols = st.columns([1, 4, 2])
                    with cols[0]:
                        st.caption(f"Q{i}")
                    with cols[1]:
                        st.write(question.get("text", "N/A"))
                    with cols[2]:
                        source = question.get("answer_source", "unknown").upper()
                        if source == "LOCAL":
                            st.write(f"🟢 {source}")
                        else:
                            st.write(f"🟡 {source}")
        else:
            st.info("No recent activity available")

    def render_full_dashboard(self) -> None:
        """Render the complete enterprise dashboard."""
        st.set_page_config(
            page_title="SecurityRAG - Enterprise Dashboard",
            page_icon="🛡️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        self.render_executive_dashboard()
        st.divider()
        self.render_ai_analytics()
        st.divider()
        self.render_threat_intelligence()
        st.divider()
        self.render_knowledge_health()
        st.divider()
        self.render_recent_activity()
