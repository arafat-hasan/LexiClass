"""Custom exceptions for LexiClass."""

from __future__ import annotations


class LexiClassError(Exception):
    """Base exception for all LexiClass errors."""
    pass


class PluginError(LexiClassError):
    """Base exception for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a requested plugin is not found in the registry."""
    pass


class PluginRegistrationError(PluginError):
    """Raised when there's an error registering a plugin."""
    pass


class PluginDependencyError(PluginError):
    """Raised when required plugin dependencies are missing."""
    pass
