from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    # Portable placeholders. Run `python setup_odtrack.py` once to rewrite
    # these to absolute paths for your checkout.
    settings.prj_dir = r'vendor/odtrack'
    settings.save_dir = r'models/odtrack'
    return settings
