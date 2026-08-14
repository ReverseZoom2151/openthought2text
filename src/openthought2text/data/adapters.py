"""Plug-in protocol and explicit registry for dataset adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable

from .manifest import DatasetManifest
from .schema import NeuralTextSample


@runtime_checkable
class DatasetAdapter(Protocol):
    """A dataset-specific importer; adapters must emit canonical examples."""

    name: str

    def build_manifest(self, source: str) -> DatasetManifest:
        """Inspect ``source`` and create its dataset manifest."""

    def iter_samples(self, source: str) -> Iterator[NeuralTextSample]:
        """Stream canonical samples from ``source`` for large datasets."""


AdapterFactory = Callable[[], DatasetAdapter]


class AdapterRegistry:
    """Small dependency-injection registry; importing an adapter never mutates it."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, name: str, factory: AdapterFactory, *, replace: bool = False) -> None:
        if not name or name.strip() != name:
            raise ValueError("adapter name must be a non-empty, trimmed string")
        if name in self._factories and not replace:
            raise KeyError(f"adapter already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> DatasetAdapter:
        try:
            adapter = self._factories[name]()
        except KeyError as error:
            available = ", ".join(self.names()) or "none"
            raise KeyError(f"unknown adapter {name!r}; available: {available}") from error
        if not isinstance(adapter, DatasetAdapter):
            raise TypeError(f"adapter factory for {name!r} did not return a DatasetAdapter")
        if adapter.name != name:
            raise ValueError(f"adapter registered as {name!r} reports name {adapter.name!r}")
        return adapter

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._factories
