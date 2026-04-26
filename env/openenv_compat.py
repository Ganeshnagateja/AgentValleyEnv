"""Small compatibility layer for OpenEnv-style inheritance."""

from __future__ import annotations

try:  # pragma: no cover - depends on optional external package shape
    from openenv.core import Environment as Environment  # type: ignore
except Exception:  # pragma: no cover - fallback path is covered by subclass tests
    try:
        from openenv_core import Environment as Environment  # type: ignore
    except Exception:
        try:
            from openenv_core.environment import Environment as Environment  # type: ignore
        except Exception:

            class Environment:
                """Minimal local base class matching the OpenEnv method contract."""

                def reset(self, *args, **kwargs):
                    raise NotImplementedError

                def step(self, *args, **kwargs):
                    raise NotImplementedError

                def state(self, *args, **kwargs):
                    raise NotImplementedError
