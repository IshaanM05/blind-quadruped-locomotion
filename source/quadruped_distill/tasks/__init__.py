"""Isaac Lab task packages: reacher (Phase 0.5), flat (Phase 1), rough (Phase 2).

Importing this package registers all of this project's own Gym task IDs (each subpackage's
`__init__.py` calls `gym.register(...)` on import), mirroring how `isaaclab_tasks` registers
its own stock tasks.
"""

from . import reacher  # noqa: F401
