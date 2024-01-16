import os
import sys
import shutil
import argparse

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Utils import utils
import Utils.yaml_config as yaml_config
import Utils.asr.multi_asr_dataset_creator_funnel_utils as multi_asr_utils


def create_output_subfolders():
    funnels_thresholds = yaml_config.cfg.get("speech_segments_funnels_thresholds")
    funnels_thresholds.sort()
    # list of tuples of the subfolders created,
    # where first element is either a tuple or a single value representing the threshold(s)
    # and the second value is the associated subfolder.
    subfolders = list()

    for i in range(len(funnels_thresholds) - 1):
        # create the folders that store speech segments whose confidences are between two thresholds.
        subfolder = os.path.join(
            yaml_config.cfg.get("output_folder"),
            f"data_funnel_gt_{funnels_thresholds[i]}_lt_{funnels_thresholds[i+1]}",
        )
        if not os.path.exists(subfolder):
            os.makedirs(subfolder, exist_ok=True)
        subfolders.append(
            ((funnels_thresholds[i], funnels_thresholds[i + 1]), subfolder)
        )
    # create the last folder that stores speech segments whose confidence are above the highest threshold.
    subfolder = os.path.join(
        yaml_config.cfg.get("output_folder"), f"data_funnel_gt_{funnels_thresholds[-1]}"
    )
    os.makedirs(subfolder, exist_ok=True)
    subfolders.append((funnels_thresholds[-1], subfolder))

    # add subfolders object to yaml config object.
    yaml_config.cfg.add(key="output_subfolders", value=subfolders)


def init():
    decoders_folderpaths = yaml_config.cfg.get("decoder_folders")
    hypothesis_level_weights = yaml_config.cfg.get("hypothesis_level_weights")

    # ensure hypotheses filenames across decoder folders match.
    hypotheses_filenames = multi_asr_utils.get_hypotheses_filenames(
        decoders_folderpaths
    )

    # ensure wavpaths match across the decoder folders' hypotheses JSON files.
    multi_asr_utils.check_wavpaths_match(decoders_folderpaths, hypotheses_filenames)

    # create output subfolders.
    create_output_subfolders()

    # create and return global input data dict.
    return multi_asr_utils.get_global_decoders_dict(
        decoders_folderpaths, hypotheses_filenames, hypothesis_level_weights
    )


def main():
    decoders_dict = init()
    # get the algorithm function object to call, whose name is specified in the yaml config.
    process = getattr(multi_asr_utils, yaml_config.cfg.get("algorithm_type"))
    # run the algorithm.
    process(decoders_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits audio files in a folder into speaker-specific diarized audio snippets."
    )
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=os.getcwd(),
        help="Path to a yaml config file.",
    )
    # parse command line arguments.
    args = parser.parse_args()

    # initialise config info from yaml file as a global Config obj called 'cfg' and validate it against the appropriate schema.
    yaml_config.init(args.config_path, yaml_config.schema_multi_asr_config)

    # setup logging to both console and logfile in the output folder.
    out_dir = yaml_config.cfg.get("output_folder")
    utils.setup_logging(
        out_dir,
        "run.log",
        console=True,
        filemode="w",
    )

    # copy yaml config file to the output folder.
    shutil.copyfile(
        args.config_path,
        f"{yaml_config.cfg.get('output_folder')}/{args.config_path.split('/')[-1]}",
    )

    # start the time profiler.
    p = utils.Profiler()
    p.start()

    main()

    p.stop()
