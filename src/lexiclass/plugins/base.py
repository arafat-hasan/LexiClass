"""Base classes and metadata for plugins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class PluginType(str, Enum):
    """Types of plugins supported by LexiClass."""
    TOKENIZER = "tokenizer"
    FEATURE_EXTRACTOR = "feature_extractor"
    CLASSIFIER = "classifier"


@dataclass
class PluginMetadata:
    """Metadata about a plugin.

    Attributes:
        name: Unique identifier for the plugin
        display_name: Human-readable name
        description: Brief description of the plugin
        plugin_type: Type of plugin (tokenizer, feature, classifier)
        dependencies: List of required packages
        optional_dependencies: List of optional packages for enhanced functionality
        supports_streaming: Whether the plugin supports streaming data
        supports_multilabel: Whether classifier supports multi-label
        default_params: Default parameters for initialization
        performance_tier: Relative speed (fast/medium/slow)
        quality_tier: Relative quality (basic/good/excellent)
    """
    name: str
    display_name: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    supports_streaming: bool = False
    supports_multilabel: bool = False
    default_params: Dict[str, Any] = field(default_factory=dict)
    performance_tier: str = "medium"  # fast, medium, slow
    quality_tier: str = "good"  # basic, good, excellent
    memory_usage: str = "medium"  # low, medium, high
    requires_pretrained: bool = False
    pretrained_models: List[str] = field(default_factory=list)

    def check_dependencies(self) -> tuple[bool, List[str]]:
        """Check if required dependencies are installed.

        Returns:
            Tuple of (all_installed, missing_packages)
        """
        import importlib.util

        # Map package names to their import names
        PACKAGE_IMPORT_MAP = {
            "scikit-learn": "sklearn",
            "beautifulsoup4": "bs4",
            "python-dateutil": "dateutil",
            "sentence-transformers": "sentence_transformers",
        }

        missing = []
        for dep in self.dependencies:
            # Parse package name (handle versions like "numpy>=1.22")
            package_name = dep.split(">=")[0].split("==")[0].split("<")[0].strip()

            # Use mapped import name if available
            import_name = PACKAGE_IMPORT_MAP.get(package_name, package_name)

            if importlib.util.find_spec(import_name) is None:
                missing.append(dep)

        return len(missing) == 0, missing


@dataclass
class PluginRegistration:
    """Complete plugin registration with metadata and factory."""
    metadata: PluginMetadata
    factory: Callable[..., Any]

    def create(self, **kwargs) -> Any:
        """Create plugin instance with parameters."""
        # Merge default params with provided kwargs
        params = {**self.metadata.default_params, **kwargs}
        return self.factory(**params)

    def is_available(self) -> bool:
        """Check if plugin can be used (dependencies installed)."""
        available, _ = self.metadata.check_dependencies()
        return available

    def get_missing_dependencies(self) -> List[str]:
        """Get list of missing dependencies."""
        _, missing = self.metadata.check_dependencies()
        return missing
