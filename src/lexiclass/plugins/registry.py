"""Enhanced plugin registry with metadata and validation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .base import PluginMetadata, PluginRegistration, PluginType
from ..exceptions import PluginNotFoundError, PluginRegistrationError, PluginDependencyError

logger = logging.getLogger(__name__)


class PluginRegistry:
    """Registry for LexiClass plugins with metadata support."""

    def __init__(self):
        self._plugins: Dict[str, PluginRegistration] = {}

    def register(
        self,
        metadata: PluginMetadata,
        factory: Callable[..., Any],
        override: bool = False,
    ) -> None:
        """Register a plugin with metadata.

        Args:
            metadata: Plugin metadata
            factory: Factory function to create plugin instance
            override: Whether to override existing plugin with same name

        Raises:
            PluginRegistrationError: If plugin already exists and override=False
        """
        if metadata.name in self._plugins and not override:
            raise PluginRegistrationError(
                f"Plugin '{metadata.name}' already registered. "
                f"Use override=True to replace."
            )

        registration = PluginRegistration(metadata=metadata, factory=factory)
        self._plugins[metadata.name] = registration
        logger.debug(f"Registered {metadata.plugin_type} plugin: {metadata.name}")

    def get(self, name: str, plugin_type: Optional[PluginType] = None) -> PluginRegistration:
        """Get plugin registration by name.

        Args:
            name: Plugin name
            plugin_type: Optional type filter

        Returns:
            PluginRegistration

        Raises:
            PluginNotFoundError: If plugin not found
        """
        if name not in self._plugins:
            available = self.list_plugins(plugin_type)
            raise PluginNotFoundError(
                f"Plugin '{name}' not found. Available plugins: {available}"
            )

        registration = self._plugins[name]

        if plugin_type and registration.metadata.plugin_type != plugin_type:
            raise PluginNotFoundError(
                f"Plugin '{name}' is a {registration.metadata.plugin_type}, "
                f"not a {plugin_type}"
            )

        return registration

    def create(
        self,
        name: str,
        plugin_type: Optional[PluginType] = None,
        **kwargs
    ) -> Any:
        """Create plugin instance.

        Args:
            name: Plugin name
            plugin_type: Optional type filter
            **kwargs: Parameters for plugin initialization

        Returns:
            Plugin instance

        Raises:
            PluginNotFoundError: If plugin not found
            PluginDependencyError: If required dependencies missing
        """
        registration = self.get(name, plugin_type)

        # Check dependencies
        if not registration.is_available():
            missing = registration.get_missing_dependencies()
            raise PluginDependencyError(
                f"Plugin '{name}' requires missing dependencies: {missing}\n"
                f"Install with: pip install {' '.join(missing)}"
            )

        return registration.create(**kwargs)

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        available_only: bool = False,
    ) -> List[str]:
        """List registered plugin names.

        Args:
            plugin_type: Filter by plugin type
            available_only: Only list plugins with dependencies installed

        Returns:
            List of plugin names
        """
        plugins = []
        for name, registration in self._plugins.items():
            if plugin_type and registration.metadata.plugin_type != plugin_type:
                continue
            if available_only and not registration.is_available():
                continue
            plugins.append(name)
        return sorted(plugins)

    def get_metadata(self, name: str) -> PluginMetadata:
        """Get plugin metadata."""
        return self.get(name).metadata

    def describe(self, name: str) -> str:
        """Get human-readable description of plugin."""
        registration = self.get(name)
        meta = registration.metadata

        status = "✓ Available" if registration.is_available() else "✗ Unavailable"
        missing = registration.get_missing_dependencies()

        lines = [
            f"Plugin: {meta.display_name} ({meta.name})",
            f"Type: {meta.plugin_type.value}",
            f"Status: {status}",
            f"Description: {meta.description}",
            f"Performance: {meta.performance_tier}",
            f"Quality: {meta.quality_tier}",
            f"Memory: {meta.memory_usage}",
            f"Streaming: {'Yes' if meta.supports_streaming else 'No'}",
        ]

        if meta.plugin_type == PluginType.CLASSIFIER:
            lines.append(f"Multi-label: {'Yes' if meta.supports_multilabel else 'No'}")

        if meta.dependencies:
            lines.append(f"Dependencies: {', '.join(meta.dependencies)}")

        if missing:
            lines.append(f"Missing: {', '.join(missing)}")
            lines.append(f"Install: pip install {' '.join(missing)}")

        if meta.default_params:
            lines.append(f"Default params: {meta.default_params}")

        if meta.pretrained_models:
            lines.append("Pre-trained models available:")
            for model in meta.pretrained_models:
                lines.append(f"  - {model}")

        return "\n".join(lines)


# Global registry instance
registry = PluginRegistry()
