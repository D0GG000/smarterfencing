"""Re-exec maintenance scripts with the app conda Python when needed."""

import os
import sys

APP_PYTHON = os.environ.get(
    "APP_PYTHON",
    "/opt/conda/envs/mmpose-env/bin/python3",
)


def reexec_if_needed():
    """Use mmpose-env Python (same as gunicorn) if the caller used system python3."""
    if not os.path.isfile(APP_PYTHON):
        return
    if os.path.realpath(sys.executable) == os.path.realpath(APP_PYTHON):
        return
    os.execv(APP_PYTHON, [APP_PYTHON, *sys.argv])


def setup_app_path():
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if app_root not in sys.path:
        sys.path.insert(0, app_root)
    return app_root
