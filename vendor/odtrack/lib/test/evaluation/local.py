from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    # Filled relative to this checkout by setup_odtrack.write_local_py()
    settings.prj_dir = r'vendor/odtrack'
    settings.save_dir = r'models/odtrack'
    return settings
