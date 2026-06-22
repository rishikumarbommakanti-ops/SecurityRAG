"""Sample knowledge documents for demo mode."""

from datetime import datetime
from typing import List, Dict


def get_demo_documents() -> List[Dict[str, object]]:
    """Return a small sample knowledge set for demo mode."""
    now = datetime.utcnow().isoformat() + "Z"
    return [
        {
            "content": (
                "MITRE ATT&CK technique T1059 covers command and script execution. "
                "This technique is often used by adversaries to execute system commands, "
                "run scripts, or invoke other tools in an interactive shell."
            ),
            "metadata": {
                "source": "MITRE ATT&CK",
                "title": "T1059: Command and Scripting Interpreter",
                "category": "Technique",
                "retrieved_at": now,
            },
        },
        {
            "content": (
                "Sigma rules are detection signatures written in a generic format. "
                "A Sigma rule typically includes log source definitions, detection patterns, "
                "and level information to identify suspicious activity."
            ),
            "metadata": {
                "source": "Sigma Rules",
                "title": "Sigma Rule Example",
                "category": "Detection",
                "retrieved_at": now,
            },
        },
        {
            "content": (
                "CISA Alerts provide advisories on active threat actors and exploited vulnerabilities. "
                "They include mitigation recommendations and detection guidance for defenders."
            ),
            "metadata": {
                "source": "CISA Alerts",
                "title": "CISA Advisory Example",
                "category": "Alert",
                "retrieved_at": now,
            },
        },
    ]
