"""Compatibility launcher for the simulation engine self-test.

The importable model lives in :mod:`network_oracle.model`.
"""

from network_oracle.model import Params, run_simulation

__all__ = ["Params", "run_simulation"]


if __name__ == "__main__":
    from network_oracle.diagnostics import main

    main()
