"""Base interface for security knowledge connectors."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional


class BaseSecurityConnector(ABC):
    """Abstract base class for all security knowledge connectors.

    Defines the interface that all connectors (MITRE, Sigma, CISA, Microsoft, LOLBas)
    must implement for retrieving and formatting security knowledge.
    """

    def __init__(self, name: str, source_url: str, refresh_interval_hours: int = 24):
        """Initialize base connector.

        Args:
            name: Human-readable connector name (e.g., "MITRE ATT&CK")
            source_url: URL to the source data
            refresh_interval_hours: How often to refresh data (default 24 hours)
        """
        self.name = name
        self.source_url = source_url
        self.refresh_interval_hours = refresh_interval_hours
        self.last_updated: Optional[datetime] = None
        self.document_count = 0
        self.is_connected = False

    @abstractmethod
    async def fetch(self) -> List[Dict[str, Any]]:
        """Fetch raw data from the source.

        Returns:
            List of document dictionaries with at minimum:
            - id: unique identifier
            - title: document title
            - content: document content
            - category: classification (e.g., "Technique", "Rule", "Alert")
            - metadata: dict with source-specific metadata
        """
        pass

    @abstractmethod
    async def validate(self) -> bool:
        """Validate connector can reach source and basic connectivity.

        Returns:
            True if connector is operational, False otherwise
        """
        pass

    @abstractmethod
    def normalize(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize fetched data to standard document format.

        Args:
            raw_data: Raw data from fetch()

        Returns:
            Normalized documents with consistent schema
        """
        pass

    def needs_refresh(self) -> bool:
        """Check if connector data needs refresh based on interval.

        Returns:
            True if last_updated is None or exceeds refresh_interval_hours
        """
        if self.last_updated is None:
            return True

        elapsed_hours = (datetime.now() - self.last_updated).total_seconds() / 3600
        return elapsed_hours >= self.refresh_interval_hours

    def get_status(self) -> Dict[str, Any]:
        """Get connector status summary.

        Returns:
            Dictionary with name, is_connected, document_count, last_updated
        """
        return {
            "name": self.name,
            "connected": self.is_connected,
            "documents": self.document_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "needs_refresh": self.needs_refresh(),
        }
