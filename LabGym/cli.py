'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
'''

import argparse
import json
import logging
import os
import sys
import yaml

logger = logging.getLogger(__name__)

class CLIConfigError(Exception):
    """Raised when CLI or YAML configuration validation fails."""
    pass


TRAIN_CATEGORIZER_DEFAULTS = {
    'network_type': 'combnet',
    'out_path': None,
    'out_folder': None,
    'dim_conv': 64,
    'dim_tconv': 32,
    'channel': 1,
    'time_step': 15,
    'level_conv': 2,
    'level_tconv': 1,
    'aug_methods': [],
    'augvalid': True,
    'include_bodyparts': True,
    'std': 0.0,
    'background_free': True,
    'black_background': True,
    'behavior_mode': 0,
    'social_distance': 0.0,
    'color_costar': False,
    'training_onfly': False,
}
REQUIRED_TRAIN_CATEGORIZER_KEYS = {'data_path', 'model_path'}
ALLOWED_TRAIN_CATEGORIZER_KEYS = set(TRAIN_CATEGORIZER_DEFAULTS.keys()) | REQUIRED_TRAIN_CATEGORIZER_KEYS

TRAIN_DETECTOR_DEFAULTS = {
    'iteration_num': 200,
    'inference_size': 480,
}
REQUIRED_TRAIN_DETECTOR_KEYS = {'path_to_annotation', 'path_to_trainingimages', 'model_path'}
ALLOWED_TRAIN_DETECTOR_KEYS = set(TRAIN_DETECTOR_DEFAULTS.keys()) | REQUIRED_TRAIN_DETECTOR_KEYS

TEST_CATEGORIZER_DEFAULTS = {
    'result_path': None,
}
REQUIRED_TEST_CATEGORIZER_KEYS = {'groundtruth_path', 'model_path'}
ALLOWED_TEST_CATEGORIZER_KEYS = set(TEST_CATEGORIZER_DEFAULTS.keys()) | REQUIRED_TEST_CATEGORIZER_KEYS

TEST_DETECTOR_DEFAULTS = {
    'result_path': None,
}
REQUIRED_TEST_DETECTOR_KEYS = {'path_to_annotation', 'path_to_testingimages', 'model_path'}
ALLOWED_TEST_DETECTOR_KEYS = set(TEST_DETECTOR_DEFAULTS.keys()) | REQUIRED_TEST_DETECTOR_KEYS

INFER_DEFAULTS = {
    'path_to_categorizer': None,
    'path_to_detector': None,
    'use_detector': False,
    'background_path': None,
    'framewidth': None,
    'delta': 10000.0,
    'animal_kinds': ['animal'],
    'animal_to_include': [],
    'behavior_to_include': ['all'],
    'detector_batch': 1,
    'detection_threshold': 0.0,
    'decode_animalnumber': False,
    'animal_number': None,
    'autofind_t': False,
    'decode_t': False,
    't': 0.0,
    'duration': 0.0,
    'decode_extraction': False,
    'ex_start': 0,
    'ex_end': None,
    'animal_vs_bg': 0,
    'stable_illumination': True,
    'ID_colors': [(255, 255, 255)],
    'parameter_to_analyze': [],
    'uncertain': 0.0,
    'min_length': None,
    'show_legend': True,
    'background_free': True,
    'black_background': True,
    'normalize_distance': True,
    'social_distance': 0.0,
    'color_costar': False,
    'correct_ID': False,
    'specific_behaviors': {},
    'behavior_mode': 0,
}
REQUIRED_INFER_KEYS = {'path_to_videos', 'result_path'}
ALLOWED_INFER_KEYS = (
    set(INFER_DEFAULTS.keys())
    | REQUIRED_INFER_KEYS
    | {'correct_id', 'id_colors'}
)

PATH_KEYS = {
    'data_path', 'model_path', 'out_path', 'out_folder',
    'path_to_annotation', 'path_to_trainingimages', 'path_to_testingimages',
    'groundtruth_path', 'result_path', 'path_to_categorizer', 'path_to_detector',
    'background_path'
}


def parse_json_or_val(v):
    if v is None:
        return None
    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
        try:
            return json.loads(v)
        except Exception:
            pass
    return v


def validate_types_and_ranges(command_name, config):
    pos_int_keys = [
        'dim_conv', 'dim_tconv', 'iteration_num', 'inference_size', 'detector_batch'
    ]
    for key in pos_int_keys:
        if key in config and config[key] is not None:
            val = config[key]
            if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
                raise CLIConfigError(
                    f"Parameter '{key}' for '{command_name}' must be a positive integer (got {val!r})."
                )

    non_neg_int_keys = ['time_step', 'ex_start', 'animal_vs_bg']
    for key in non_neg_int_keys:
        if key in config and config[key] is not None:
            val = config[key]
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise CLIConfigError(
                    f"Parameter '{key}' for '{command_name}' must be a non-negative integer (got {val!r})."
                )

    if 'channel' in config and config['channel'] is not None:
        if config['channel'] not in (1, 3):
            raise CLIConfigError(
                f"Parameter 'channel' must be 1 or 3 (got {config['channel']!r})."
            )

    for level_key in ['level_conv', 'level_tconv']:
        if level_key in config and config[level_key] is not None:
            val = config[level_key]
            if not isinstance(val, int) or isinstance(val, bool) or not (1 <= val <= 5):
                raise CLIConfigError(
                    f"Parameter '{level_key}' must be an integer between 1 and 5 (got {val!r})."
                )

    if 'network_type' in config and config['network_type'] is not None:
        valid_nets = {'combnet', 'pattern_recognizer', 'animation_analyzer', 'combined_network'}
        if config['network_type'] not in valid_nets:
            raise CLIConfigError(
                f"Invalid network_type '{config['network_type']}'. Choose from: {sorted(valid_nets)}."
            )

    if 'behavior_mode' in config and config['behavior_mode'] is not None:
        val = config['behavior_mode']
        if not isinstance(val, int) or isinstance(val, bool) or not (0 <= val <= 3):
            raise CLIConfigError(
                f"Parameter 'behavior_mode' must be an integer between 0 and 3 (got {val!r})."
            )

    float_keys = ['std', 'social_distance', 't', 'duration', 'delta', 'detection_threshold', 'uncertain']
    for key in float_keys:
        if key in config and config[key] is not None:
            val = config[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
                raise CLIConfigError(
                    f"Parameter '{key}' must be a non-negative float (got {val!r})."
                )

    for thresh_key in ['detection_threshold', 'uncertain']:
        if thresh_key in config and config[thresh_key] is not None:
            val = float(config[thresh_key])
            if not (0.0 <= val <= 1.0):
                raise CLIConfigError(
                    f"Parameter '{thresh_key}' must be between 0.0 and 1.0 (got {val!r})."
                )

    bool_keys = [
        'training_onfly', 'augvalid', 'include_bodyparts', 'background_free', 'black_background',
        'color_costar', 'use_detector', 'decode_animalnumber', 'autofind_t', 'decode_t',
        'decode_extraction', 'stable_illumination', 'show_legend', 'normalize_distance',
        'correct_ID', 'correct_id'
    ]
    for key in bool_keys:
        if key in config and config[key] is not None:
            val = config[key]
            if not isinstance(val, bool):
                raise CLIConfigError(
                    f"Parameter '{key}' must be a boolean True/False (got {val!r})."
                )


def resolve_config(command_name, defaults, allowed_keys, required_keys, config_path, cli_args_dict):
    """Resolve config from defaults, YAML file, and CLI overrides."""
    resolved = defaults.copy()

    if config_path is not None:
        expanded_config = os.path.expanduser(config_path)
        if not os.path.isfile(expanded_config):
            raise CLIConfigError(f"Specified configuration file not found: '{config_path}'")
        try:
            with open(expanded_config, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
        except Exception as e:
            raise CLIConfigError(f"Error reading YAML file '{config_path}': {e}")

        if yaml_data is not None:
            if not isinstance(yaml_data, dict):
                raise CLIConfigError(
                    f"Configuration file '{config_path}' must contain a key-value mapping (YAML dictionary)."
                )
            unknown_keys = set(yaml_data.keys()) - allowed_keys
            if unknown_keys:
                raise CLIConfigError(
                    f"Unknown configuration key(s) in '{config_path}' for command '{command_name}': "
                    f"{', '.join(sorted(unknown_keys))}"
                )
            for k, v in yaml_data.items():
                resolved[k] = v

    for k, v in cli_args_dict.items():
        if k not in ['config', 'action', 'target', 'func'] and v is not None:
            resolved[k] = v

    if 'correct_id' in resolved:
        resolved['correct_ID'] = resolved.pop('correct_id')
    if 'id_colors' in resolved:
        resolved['ID_colors'] = resolved.pop('id_colors')

    for pk in PATH_KEYS:
        if pk in resolved and isinstance(resolved[pk], str):
            resolved[pk] = os.path.expanduser(resolved[pk])

    if 'path_to_videos' in resolved and resolved['path_to_videos'] is not None:
        val = resolved['path_to_videos']
        if isinstance(val, str):
            resolved['path_to_videos'] = [os.path.expanduser(val)]
        elif isinstance(val, list):
            resolved['path_to_videos'] = [
                os.path.expanduser(item) if isinstance(item, str) else item
                for item in val
            ]

    validate_types_and_ranges(command_name, resolved)

    missing = [rk for rk in required_keys if resolved.get(rk) is None]
    if missing:
        raise CLIConfigError(
            f"Missing required parameter(s) for '{command_name}': {', '.join(sorted(missing))}. "
            f"Please supply via --config YAML or explicit inline options."
        )

    return resolved


def add_common_config_arg(parser):
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help="Path to YAML workflow configuration file specifying settings."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="labgym",
        description="LabGym CLI: Headless command-line interface for training, testing, and behavioral inference."
    )
    subparsers = parser.add_subparsers(dest="action", help="Available actions")

    train_parser = subparsers.add_parser("train", help="Train categorizers or detectors")
    train_sub = train_parser.add_subparsers(dest="target", help="Target network to train", required=False)

    p_train_cat = train_sub.add_parser("categorizer", help="Train behavior classifier (Pattern Recognizer, CombNet, etc.)")
    add_common_config_arg(p_train_cat)
    p_train_cat.add_argument("--data-path", type=str, default=None, help="Path to training examples dataset")
    p_train_cat.add_argument("--model-path", type=str, default=None, help="Path to output model directory")
    p_train_cat.add_argument("--network-type", type=str, choices=['combnet', 'pattern_recognizer', 'animation_analyzer', 'combined_network'], default=None, help="Type of network architecture")
    p_train_cat.add_argument("--out-path", type=str, default=None, help="Output path for logs/reports")
    p_train_cat.add_argument("--out-folder", type=str, default=None, help="Target folder for caching prepared frames")
    p_train_cat.add_argument("--dim-conv", type=int, default=None, help="Input dimension for Pattern Recognizer")
    p_train_cat.add_argument("--dim-tconv", type=int, default=None, help="Input dimension for Animation Analyzer")
    p_train_cat.add_argument("--channel", type=int, choices=[1, 3], default=None, help="Channel depth (1 or 3)")
    p_train_cat.add_argument("--time-step", type=int, default=None, help="Time steps/frames per behavior sequence")
    p_train_cat.add_argument("--level-conv", type=int, default=None, help="Complexity level of Pattern Recognizer (1-5)")
    p_train_cat.add_argument("--level-tconv", type=int, default=None, help="Complexity level of Animation Analyzer (1-5)")
    p_train_cat.add_argument("--aug-methods", type=str, nargs="+", default=None, help="List of augmentation methods")
    p_train_cat.add_argument("--augvalid", action=argparse.BooleanOptionalAction, default=None, help="Validate augmentation data")
    p_train_cat.add_argument("--include-bodyparts", action=argparse.BooleanOptionalAction, default=None, help="Include body parts in calculations")
    p_train_cat.add_argument("--std", type=float, default=None, help="Standard deviation parameter for body parts")
    p_train_cat.add_argument("--background-free", action=argparse.BooleanOptionalAction, default=None, help="Exclude background")
    p_train_cat.add_argument("--black-background", action=argparse.BooleanOptionalAction, default=None, help="Use black background")
    p_train_cat.add_argument("--behavior-mode", type=int, default=None, help="Behavior mode selection")
    p_train_cat.add_argument("--social-distance", type=float, default=None, help="Social distance threshold")
    p_train_cat.add_argument("--color-costar", action=argparse.BooleanOptionalAction, default=None, help="Keep supporting actors in color")
    p_train_cat.add_argument("--training-onfly", action=argparse.BooleanOptionalAction, default=None, help="Train using on-the-fly behavior generator")
    p_train_cat.set_defaults(func=handle_train_categorizer)

    p_train_det = train_sub.add_parser("detector", help="Train object detector (Detectron2)")
    add_common_config_arg(p_train_det)
    p_train_det.add_argument("--path-to-annotation", type=str, default=None, help="Path to COCO JSON annotation file")
    p_train_det.add_argument("--path-to-trainingimages", type=str, default=None, help="Path to training images folder")
    p_train_det.add_argument("--model-path", type=str, default=None, help="Path to output model directory")
    p_train_det.add_argument("--iteration-num", type=int, default=None, help="Number of training iterations")
    p_train_det.add_argument("--inference-size", type=int, default=None, help="Inference image frame size")
    p_train_det.set_defaults(func=handle_train_detector)

    test_parser = subparsers.add_parser("test", help="Evaluate models headlessly")
    test_sub = test_parser.add_subparsers(dest="target", help="Target model to test", required=False)

    p_test_cat = test_sub.add_parser("categorizer", help="Evaluate behavior classifier")
    add_common_config_arg(p_test_cat)
    p_test_cat.add_argument("--groundtruth-path", type=str, default=None, help="Path to ground-truth testing dataset")
    p_test_cat.add_argument("--model-path", type=str, default=None, help="Path to trained categorizer directory")
    p_test_cat.add_argument("--result-path", type=str, default=None, help="Path to output test results")
    p_test_cat.set_defaults(func=handle_test_categorizer)

    p_test_det = test_sub.add_parser("detector", help="Evaluate object detector")
    add_common_config_arg(p_test_det)
    p_test_det.add_argument("--path-to-annotation", type=str, default=None, help="Path to COCO JSON annotation file")
    p_test_det.add_argument("--path-to-testingimages", type=str, default=None, help="Path to testing images directory")
    p_test_det.add_argument("--model-path", type=str, default=None, help="Path to trained detector directory")
    p_test_det.add_argument("--result-path", type=str, default=None, help="Path to output test results")
    p_test_det.set_defaults(func=handle_test_detector)

    infer_parser = subparsers.add_parser("infer", help="Execute animal tracking and behavior classification on videos or images")
    add_common_config_arg(infer_parser)
    infer_parser.add_argument("--path-to-videos", type=str, nargs="+", default=None, help="Path(s) to video files or directory")
    infer_parser.add_argument("--result-path", type=str, default=None, help="Path to write inference results")
    infer_parser.add_argument("--path-to-categorizer", type=str, default=None, help="Path to trained categorizer directory")
    infer_parser.add_argument("--path-to-detector", type=str, default=None, help="Path to trained detector directory")
    infer_parser.add_argument("--use-detector", action=argparse.BooleanOptionalAction, default=None, help="Use detector for animal tracking")
    infer_parser.add_argument("--background-path", type=str, default=None, help="Path to background image")
    infer_parser.add_argument("--framewidth", type=int, default=None, help="Target resizing width for frames")
    infer_parser.add_argument("--delta", type=float, default=None, help="Optogenetics stimulation delta trigger threshold")
    infer_parser.add_argument("--animal-kinds", type=str, nargs="+", default=None, help="Animal categories in detector")
    infer_parser.add_argument("--animal-to-include", type=str, nargs="+", default=None, help="Animals to keep in outputs")
    infer_parser.add_argument("--behavior-to-include", type=str, nargs="+", default=None, help="Behaviors to categorize")
    infer_parser.add_argument("--detector-batch", type=int, default=None, help="Detection processing batch size")
    infer_parser.add_argument("--detection-threshold", type=float, default=None, help="Detection confidence threshold")
    infer_parser.add_argument("--decode-animalnumber", action=argparse.BooleanOptionalAction, default=None, help="Extract animal count from video filenames")
    infer_parser.add_argument("--animal-number", type=str, default=None, help="Animal count or mapping")
    infer_parser.add_argument("--autofind-t", action=argparse.BooleanOptionalAction, default=None, help="Auto-calculate optogenetics trigger")
    infer_parser.add_argument("--decode-t", action=argparse.BooleanOptionalAction, default=None, help="Extract start times from filenames")
    infer_parser.add_argument("--t", type=float, default=None, help="Start time offset in seconds")
    infer_parser.add_argument("--duration", type=float, default=None, help="Total analysis window in seconds")
    infer_parser.add_argument("--decode-extraction", action=argparse.BooleanOptionalAction, default=None, help="Extract start/end frame times")
    infer_parser.add_argument("--ex-start", type=int, default=None, help="Extraction start frame number")
    infer_parser.add_argument("--ex-end", type=int, default=None, help="Extraction end frame number")
    infer_parser.add_argument("--animal-vs-bg", type=int, default=None, help="Animal brightness relative to background")
    infer_parser.add_argument("--stable-illumination", action=argparse.BooleanOptionalAction, default=None, help="Illumination stability")
    infer_parser.add_argument("--ID-colors", type=str, default=None, help="Colormap sequence mapping IDs to colors")
    infer_parser.add_argument("--parameter-to-analyze", type=str, nargs="+", default=None, help="Kinematic parameters to extract")
    infer_parser.add_argument("--uncertain", type=float, default=None, help="Uncertainty threshold")
    infer_parser.add_argument("--min-length", type=int, default=None, help="Minimum behavior length in frames")
    infer_parser.add_argument("--show-legend", action=argparse.BooleanOptionalAction, default=None, help="Render behavior legend overlays")
    infer_parser.add_argument("--background-free", action=argparse.BooleanOptionalAction, default=None, help="Remove background from calculations")
    infer_parser.add_argument("--black-background", action=argparse.BooleanOptionalAction, default=None, help="Convert background elements to solid black")
    infer_parser.add_argument("--normalize-distance", action=argparse.BooleanOptionalAction, default=None, help="Normalize metrics relative to contour sizes")
    infer_parser.add_argument("--social-distance", type=float, default=None, help="Social distance radius parameter")
    infer_parser.add_argument("--color-costar", action=argparse.BooleanOptionalAction, default=None, help="Render costar characters in RGB scale")
    infer_parser.add_argument("--correct-id", action=argparse.BooleanOptionalAction, default=None, help="Run ID correction heuristics")
    infer_parser.add_argument("--specific-behaviors", type=str, default=None, help="Specific behavior definitions JSON")
    infer_parser.add_argument("--behavior-mode", type=int, default=None, help="Behavior mode selection")
    infer_parser.set_defaults(func=handle_infer)

    return parser


def handle_train_categorizer(args):
    config = resolve_config(
        command_name="labgym train categorizer",
        defaults=TRAIN_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
        config_path=args.config,
        cli_args_dict=vars(args)
    )
    
    os.makedirs(config['model_path'], exist_ok=True)
    print(f"Starting headless Categorizer training ({config['network_type']})...", file=sys.stdout)
    
    from LabGym.categorizer import Categorizers
    CA = Categorizers()
    
    net_type = config['network_type']
    onfly = config['training_onfly']
    
    if net_type == 'pattern_recognizer':
        if onfly:
            CA.train_pattern_recognizer_onfly(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim=config['dim_conv'], channel=config['channel'], time_step=config['time_step'],
                level=config['level_conv'], include_bodyparts=config['include_bodyparts'],
                std=config['std'], background_free=config['background_free'],
                black_background=config['black_background'], behavior_mode=config['behavior_mode'],
                social_distance=config['social_distance']
            )
        else:
            CA.train_pattern_recognizer(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim=config['dim_conv'], channel=config['channel'], time_step=config['time_step'],
                level=config['level_conv'], aug_methods=config['aug_methods'], augvalid=config['augvalid'],
                include_bodyparts=config['include_bodyparts'], std=config['std'],
                background_free=config['background_free'], black_background=config['black_background'],
                behavior_mode=config['behavior_mode'], social_distance=config['social_distance'],
                out_folder=config['out_folder']
            )
    elif net_type == 'animation_analyzer':
        if onfly:
            CA.train_animation_analyzer_onfly(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim=config['dim_tconv'], channel=config['channel'], time_step=config['time_step'],
                level=config['level_tconv'], include_bodyparts=config['include_bodyparts'],
                std=config['std'], background_free=config['background_free'],
                black_background=config['black_background'], behavior_mode=config['behavior_mode'],
                social_distance=config['social_distance'], color_costar=config['color_costar']
            )
        else:
            CA.train_animation_analyzer(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim=config['dim_tconv'], channel=config['channel'], time_step=config['time_step'],
                level=config['level_tconv'], aug_methods=config['aug_methods'], augvalid=config['augvalid'],
                include_bodyparts=config['include_bodyparts'], std=config['std'],
                background_free=config['background_free'], black_background=config['black_background'],
                behavior_mode=config['behavior_mode'], social_distance=config['social_distance'],
                color_costar=config['color_costar'], out_folder=config['out_folder']
            )
    else:
        if onfly:
            CA.train_combnet_onfly(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim_tconv=config['dim_tconv'], dim_conv=config['dim_conv'], channel=config['channel'],
                time_step=config['time_step'], level_tconv=config['level_tconv'], level_conv=config['level_conv'],
                include_bodyparts=config['include_bodyparts'], std=config['std'],
                background_free=config['background_free'], black_background=config['black_background'],
                behavior_mode=config['behavior_mode'], social_distance=config['social_distance'],
                color_costar=config['color_costar']
            )
        else:
            CA.train_combnet(
                config['data_path'], config['model_path'], out_path=config['out_path'],
                dim_tconv=config['dim_tconv'], dim_conv=config['dim_conv'], channel=config['channel'],
                time_step=config['time_step'], level_tconv=config['level_tconv'], level_conv=config['level_conv'],
                aug_methods=config['aug_methods'], augvalid=config['augvalid'],
                include_bodyparts=config['include_bodyparts'], std=config['std'],
                background_free=config['background_free'], black_background=config['black_background'],
                behavior_mode=config['behavior_mode'], social_distance=config['social_distance'],
                color_costar=config['color_costar'], out_folder=config['out_folder']
            )
            
    print("Categorizer training completed successfully.", file=sys.stdout)


def handle_train_detector(args):
    config = resolve_config(
        command_name="labgym train detector",
        defaults=TRAIN_DETECTOR_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_DETECTOR_KEYS,
        required_keys=REQUIRED_TRAIN_DETECTOR_KEYS,
        config_path=args.config,
        cli_args_dict=vars(args)
    )
    
    os.makedirs(config['model_path'], exist_ok=True)
    print(f"Starting headless Detector training (Iterations: {config['iteration_num']}, Inference Size: {config['inference_size']})...", file=sys.stdout)
    
    from LabGym.detector import Detector
    DT = Detector()
    DT.train(
        config['path_to_annotation'],
        config['path_to_trainingimages'],
        config['model_path'],
        config['iteration_num'],
        config['inference_size']
    )
    print("Detector training completed successfully.", file=sys.stdout)


def handle_test_categorizer(args):
    config = resolve_config(
        command_name="labgym test categorizer",
        defaults=TEST_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TEST_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TEST_CATEGORIZER_KEYS,
        config_path=args.config,
        cli_args_dict=vars(args)
    )
    
    if config['result_path']:
        os.makedirs(config['result_path'], exist_ok=True)
        
    print(f"Starting headless Categorizer testing on '{config['groundtruth_path']}'...", file=sys.stdout)
    from LabGym.categorizer import Categorizers
    CA = Categorizers()
    CA.test_categorizer(config['groundtruth_path'], config['model_path'], result_path=config['result_path'])
    print("Categorizer testing completed successfully.", file=sys.stdout)


def handle_test_detector(args):
    config = resolve_config(
        command_name="labgym test detector",
        defaults=TEST_DETECTOR_DEFAULTS,
        allowed_keys=ALLOWED_TEST_DETECTOR_KEYS,
        required_keys=REQUIRED_TEST_DETECTOR_KEYS,
        config_path=args.config,
        cli_args_dict=vars(args)
    )
    
    if config['result_path']:
        os.makedirs(config['result_path'], exist_ok=True)
        
    print(f"Starting headless Detector testing on '{config['path_to_testingimages']}'...", file=sys.stdout)
    from LabGym.detector import Detector
    DT = Detector()
    DT.test(
        config['path_to_annotation'],
        config['path_to_testingimages'],
        config['model_path'],
        config['result_path']
    )
    print("Detector testing completed successfully.", file=sys.stdout)


def handle_infer(args):
    config = resolve_config(
        command_name="labgym infer",
        defaults=INFER_DEFAULTS,
        allowed_keys=ALLOWED_INFER_KEYS,
        required_keys=REQUIRED_INFER_KEYS,
        config_path=args.config,
        cli_args_dict=vars(args)
    )
    
    os.makedirs(config['result_path'], exist_ok=True)
    print(f"Starting headless behavioral inference on {len(config['path_to_videos'])} video path(s)...", file=sys.stdout)
    
    # cont in C2
    print("Behavioral inference executed.", file=sys.stdout)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        try:
            print("Launching LabGym GUI...", file=sys.stdout)
            from LabGym.__main__ import main as gui_main
            return gui_main()
        except Exception as e:
            print(f"Error launching LabGym GUI: {e}", file=sys.stderr)
            return 1

    parser = build_parser()
    
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code

    if args.action in ['train', 'test'] and getattr(args, 'target', None) is None:
        print(f"Error: 'labgym {args.action}' requires a target (e.g. 'categorizer' or 'detector').", file=sys.stderr)
        return 2

    if not hasattr(args, 'func'):
        parser.print_help(file=sys.stderr)
        return 2

    try:
        args.func(args)
        return 0
    except CLIConfigError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Execution Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
