from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = '/data/uetrack_src_20260825/UETrack'
    settings.save_dir = '/data/uetrack_output_20260825'
    settings.results_path = '/data/uetrack_output_20260825/test/tracking_results'
    settings.network_path = '/data/uetrack_output_20260825/test/networks'
    settings.segmentation_path = '/data/uetrack_output_20260825/test/segmentation_results'
    settings.result_plot_path = '/data/uetrack_output_20260825/test/result_plots'
    return settings
