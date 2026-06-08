"""Package init: configure JAX before any submodule imports it."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
