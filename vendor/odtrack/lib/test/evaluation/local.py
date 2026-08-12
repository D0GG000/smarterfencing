from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = r'/home/jordan/fencing-mmpose-dev3/app/vendor/odtrack'
    settings.save_dir = r'/home/jordan/fencing-mmpose-dev3/app/models/odtrack'
    return settings
