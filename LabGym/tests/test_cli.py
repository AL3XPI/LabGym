import json
import os
import sys
import pytest
import yaml
from unittest.mock import patch, MagicMock

from LabGym.cli import (
    build_parser,
    resolve_config,
    CLIConfigError,
    main,
    load_categorizer_metadata,
    load_detector_metadata,
    resolve_media_paths,
    TRAIN_CATEGORIZER_DEFAULTS,
    ALLOWED_TRAIN_CATEGORIZER_KEYS,
    REQUIRED_TRAIN_CATEGORIZER_KEYS,
    TRAIN_DETECTOR_DEFAULTS,
    ALLOWED_TRAIN_DETECTOR_KEYS,
    REQUIRED_TRAIN_DETECTOR_KEYS,
    TEST_CATEGORIZER_DEFAULTS,
    ALLOWED_TEST_CATEGORIZER_KEYS,
    REQUIRED_TEST_CATEGORIZER_KEYS,
    TEST_DETECTOR_DEFAULTS,
    ALLOWED_TEST_DETECTOR_KEYS,
    REQUIRED_TEST_DETECTOR_KEYS,
    INFER_DEFAULTS,
    ALLOWED_INFER_KEYS,
    REQUIRED_INFER_KEYS,
)


def test_command_routing_train_categorizer():
    parser = build_parser()
    args = parser.parse_args([
        "train", "categorizer",
        "--data-path", "/path/to/data",
        "--model-path", "/path/to/model"
    ])
    assert args.action == "train"
    assert args.target == "categorizer"
    assert args.data_path == "/path/to/data"
    assert args.model_path == "/path/to/model"


def test_command_routing_train_detector():
    parser = build_parser()
    args = parser.parse_args([
        "train", "detector",
        "--path-to-annotation", "annot.json",
        "--path-to-trainingimages", "imgs/",
        "--model-path", "det_model/"
    ])
    assert args.action == "train"
    assert args.target == "detector"
    assert args.path_to_annotation == "annot.json"
    assert args.path_to_trainingimages == "imgs/"
    assert args.model_path == "det_model/"


def test_command_routing_test_categorizer():
    parser = build_parser()
    args = parser.parse_args([
        "test", "categorizer",
        "--groundtruth-path", "gt/",
        "--model-path", "cat_model/"
    ])
    assert args.action == "test"
    assert args.target == "categorizer"
    assert args.groundtruth_path == "gt/"
    assert args.model_path == "cat_model/"


def test_command_routing_test_detector():
    parser = build_parser()
    args = parser.parse_args([
        "test", "detector",
        "--path-to-annotation", "test_annot.json",
        "--path-to-testingimages", "test_imgs/",
        "--model-path", "det_model/"
    ])
    assert args.action == "test"
    assert args.target == "detector"
    assert args.path_to_annotation == "test_annot.json"
    assert args.path_to_testingimages == "test_imgs/"
    assert args.model_path == "det_model/"


def test_command_routing_infer():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--path-to-videos", "vid1.mp4", "vid2.mp4",
        "--result-path", "out/"
    ])
    assert args.action == "infer"
    assert args.path_to_videos == ["vid1.mp4", "vid2.mp4"]
    assert args.result_path == "out/"


def test_missing_target_fails():
    exit_code = main(["train"])
    assert exit_code == 2

    exit_code_test = main(["test"])
    assert exit_code_test == 2


def test_precedence_defaults_only():
    cli_args = {
        'data_path': '/my/data',
        'model_path': '/my/model',
    }
    resolved = resolve_config(
        command_name="labgym train categorizer",
        defaults=TRAIN_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
        config_path=None,
        cli_args_dict=cli_args
    )
    assert resolved['dim_conv'] == 64
    assert resolved['channel'] == 1
    assert resolved['level_conv'] == 2
    assert resolved['background_free'] is True
    assert resolved['data_path'] == '/my/data'
    assert resolved['model_path'] == '/my/model'


def test_precedence_yaml_overrides_defaults(tmp_path):
    config_file = tmp_path / "train_cat.yaml"
    yaml_content = {
        'data_path': '/yaml/data',
        'model_path': '/yaml/model',
        'dim_conv': 128,
        'channel': 3,
        'background_free': False,
    }
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    cli_args = {}
    resolved = resolve_config(
        command_name="labgym train categorizer",
        defaults=TRAIN_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
        config_path=str(config_file),
        cli_args_dict=cli_args
    )

    assert resolved['dim_conv'] == 128
    assert resolved['channel'] == 3
    assert resolved['background_free'] is False
    assert resolved['data_path'] == '/yaml/data'
    assert resolved['dim_tconv'] == 32
    assert resolved['time_step'] == 15


def test_precedence_inline_overrides_yaml(tmp_path):
    config_file = tmp_path / "train_cat.yaml"
    yaml_content = {
        'data_path': '/yaml/data',
        'model_path': '/yaml/model',
        'dim_conv': 128,
        'channel': 3,
        'background_free': False,
    }
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    cli_args = {
        'dim_conv': 256,
        'background_free': True,
    }
    resolved = resolve_config(
        command_name="labgym train categorizer",
        defaults=TRAIN_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
        config_path=str(config_file),
        cli_args_dict=cli_args
    )

    assert resolved['dim_conv'] == 256
    assert resolved['background_free'] is True
    assert resolved['channel'] == 3
    assert resolved['data_path'] == '/yaml/data'


def test_absent_inline_flags_never_overwrite_yaml(tmp_path):
    config_file = tmp_path / "infer_job.yaml"
    yaml_content = {
        'path_to_videos': ['/yaml/vids'],
        'result_path': '/yaml/results',
        'use_detector': True,
        'delta': 500.0,
    }
    with open(config_file, 'w') as f:
        yaml.dump(yaml_content, f)

    cli_args = {
        'use_detector': None,
        'delta': None,
        'path_to_videos': None,
        'result_path': None,
    }
    resolved = resolve_config(
        command_name="labgym infer",
        defaults=INFER_DEFAULTS,
        allowed_keys=ALLOWED_INFER_KEYS,
        required_keys=REQUIRED_INFER_KEYS,
        config_path=str(config_file),
        cli_args_dict=cli_args
    )

    assert resolved['use_detector'] is True
    assert resolved['delta'] == 500.0
    assert resolved['result_path'] == '/yaml/results'


def test_paired_boolean_flags_positive():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--path-to-videos", "vid.mp4",
        "--result-path", "out/",
        "--use-detector"
    ])
    assert args.use_detector is True


def test_paired_boolean_flags_negative():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--path-to-videos", "vid.mp4",
        "--result-path", "out/",
        "--no-use-detector"
    ])
    assert args.use_detector is False


def test_paired_boolean_flags_unspecified_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--path-to-videos", "vid.mp4",
        "--result-path", "out/"
    ])
    assert args.use_detector is None


def test_paired_boolean_flag_overrides_yaml_false_with_true(tmp_path):
    cfg = tmp_path / "config.yaml"
    with open(cfg, 'w') as f:
        yaml.dump({'path_to_videos': 'vid.mp4', 'result_path': 'out/', 'use_detector': False}, f)

    resolved = resolve_config(
        command_name="labgym infer",
        defaults=INFER_DEFAULTS,
        allowed_keys=ALLOWED_INFER_KEYS,
        required_keys=REQUIRED_INFER_KEYS,
        config_path=str(cfg),
        cli_args_dict={'use_detector': True}
    )
    assert resolved['use_detector'] is True


def test_paired_boolean_flag_overrides_yaml_true_with_false(tmp_path):
    cfg = tmp_path / "config.yaml"
    with open(cfg, 'w') as f:
        yaml.dump({'path_to_videos': 'vid.mp4', 'result_path': 'out/', 'use_detector': True}, f)

    resolved = resolve_config(
        command_name="labgym infer",
        defaults=INFER_DEFAULTS,
        allowed_keys=ALLOWED_INFER_KEYS,
        required_keys=REQUIRED_INFER_KEYS,
        config_path=str(cfg),
        cli_args_dict={'use_detector': False}
    )
    assert resolved['use_detector'] is False


def test_missing_config_file_raises_error():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path="/nonexistent/path/to/missing_config.yaml",
            cli_args_dict={}
        )
    assert "Specified configuration file not found" in str(exc_info.value)


def test_unknown_yaml_key_raises_error(tmp_path):
    bad_cfg = tmp_path / "bad.yaml"
    with open(bad_cfg, 'w') as f:
        yaml.dump({
            'data_path': '/data',
            'model_path': '/model',
            'unknown_bad_key': 123
        }, f)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=str(bad_cfg),
            cli_args_dict={}
        )
    assert "Unknown configuration key(s)" in str(exc_info.value)
    assert "unknown_bad_key" in str(exc_info.value)


def test_missing_required_keys_raises_error():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=None,
            cli_args_dict={'data_path': '/some/data'}
        )
    assert "Missing required parameter(s)" in str(exc_info.value)
    assert "model_path" in str(exc_info.value)


def test_invalid_integer_bounds_raises_error():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train detector",
            defaults=TRAIN_DETECTOR_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_DETECTOR_KEYS,
            required_keys=REQUIRED_TRAIN_DETECTOR_KEYS,
            config_path=None,
            cli_args_dict={
                'path_to_annotation': 'a.json',
                'path_to_trainingimages': 'imgs/',
                'model_path': 'det/',
                'iteration_num': -100
            }
        )
    assert "Parameter 'iteration_num' for 'labgym train detector' must be a positive integer" in str(exc_info.value)


def test_invalid_channel_raises_error():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=None,
            cli_args_dict={
                'data_path': '/data',
                'model_path': '/model',
                'channel': 2
            }
        )
    assert "Parameter 'channel' must be 1 or 3" in str(exc_info.value)


def test_invalid_network_type_raises_error():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=None,
            cli_args_dict={
                'data_path': '/data',
                'model_path': '/model',
                'network_type': 'nonexistent_network'
            }
        )
    assert "Invalid network_type 'nonexistent_network'" in str(exc_info.value)


def test_detector_training_does_not_require_categorizer_data():
    cli_args = {
        'path_to_annotation': 'annot.json',
        'path_to_trainingimages': 'images/',
        'model_path': 'models/detector',
    }
    resolved = resolve_config(
        command_name="labgym train detector",
        defaults=TRAIN_DETECTOR_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_DETECTOR_KEYS,
        required_keys=REQUIRED_TRAIN_DETECTOR_KEYS,
        config_path=None,
        cli_args_dict=cli_args
    )
    assert resolved['path_to_annotation'] == 'annot.json'
    assert resolved['iteration_num'] == 200
    assert 'data_path' not in resolved


def test_user_path_expansion():
    cli_args = {
        'data_path': '~/my_dataset',
        'model_path': '~/my_model',
    }
    resolved = resolve_config(
        command_name="labgym train categorizer",
        defaults=TRAIN_CATEGORIZER_DEFAULTS,
        allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
        required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
        config_path=None,
        cli_args_dict=cli_args
    )
    expected_home = os.path.expanduser('~')
    assert resolved['data_path'].startswith(expected_home)
    assert resolved['model_path'].startswith(expected_home)
    assert not resolved['data_path'].startswith('~')


def test_cli_main_missing_config_returns_non_zero():
    exit_code = main(["train", "categorizer", "-c", "missing_file_does_not_exist.yaml"])
    assert exit_code == 2


def test_cli_main_invalid_args_returns_code_2():
    exit_code = main(["infer", "--unknown-flag-1234"])
    assert exit_code == 2


def test_booleans_rejected_for_integer_and_numeric_fields():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=None,
            cli_args_dict={'data_path': 'd', 'model_path': 'm', 'channel': True}
        )
    assert "Parameter 'channel' must be 1 or 3" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=None,
            cli_args_dict={'data_path': 'd', 'model_path': 'm', 'dim_conv': True}
        )
    assert "Parameter 'dim_conv' for 'labgym train categorizer' must be a positive integer" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'framewidth': True}
        )
    assert "Parameter 'framewidth' for 'labgym infer' must be a positive integer" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'min_length': False}
        )
    assert "Parameter 'min_length' for 'labgym infer' must be a positive integer" in str(exc_info.value)


def test_negative_values_rejected_for_framewidth_and_min_length():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'framewidth': -640}
        )
    assert "Parameter 'framewidth' for 'labgym infer' must be a positive integer" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'min_length': -5}
        )
    assert "Parameter 'min_length' for 'labgym infer' must be a positive integer" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'ex_end': -1}
        )
    assert "Parameter 'ex_end' for 'labgym infer' must be a non-negative integer" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['v.mp4'], 'result_path': 'r', 'ex_start': 100, 'ex_end': 50}
        )
    assert "Parameter 'ex_end' (50) must be greater than or equal to 'ex_start' (100)" in str(exc_info.value)


def test_strings_rejected_where_lists_are_expected(tmp_path):
    cfg = tmp_path / "bad_lists.yaml"
    with open(cfg, 'w') as f:
        yaml.dump({
            'path_to_videos': ['v.mp4'],
            'result_path': 'r',
            'animal_kinds': "mouse"
        }, f)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=str(cfg),
            cli_args_dict={}
        )
    assert "Parameter 'animal_kinds' for 'labgym infer' must be a list of strings" in str(exc_info.value)

    with open(cfg, 'w') as f:
        yaml.dump({
            'data_path': 'd',
            'model_path': 'm',
            'aug_methods': "rotation"
        }, f)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym train categorizer",
            defaults=TRAIN_CATEGORIZER_DEFAULTS,
            allowed_keys=ALLOWED_TRAIN_CATEGORIZER_KEYS,
            required_keys=REQUIRED_TRAIN_CATEGORIZER_KEYS,
            config_path=str(cfg),
            cli_args_dict={}
        )
    assert "Parameter 'aug_methods' for 'labgym train categorizer' must be a list of strings" in str(exc_info.value)

    with open(cfg, 'w') as f:
        yaml.dump({
            'path_to_videos': ['v.mp4'],
            'result_path': 'r',
            'animal_kinds': [123]
        }, f)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=str(cfg),
            cli_args_dict={}
        )
    assert "Items in 'animal_kinds' for 'labgym infer' must be strings" in str(exc_info.value)


def test_infer_rejects_model_preprocessing_settings(tmp_path):
    cfg = tmp_path / "bad_infer.yaml"
    for forbidden_key in ['background_free', 'black_background', 'behavior_mode', 'animal_vs_bg', 'social_distance', 'color_costar']:
        with open(cfg, 'w') as f:
            yaml.dump({
                'path_to_videos': ['v.mp4'],
                'result_path': 'r',
                forbidden_key: True if 'background' in forbidden_key or 'color' in forbidden_key else 0
            }, f)
        with pytest.raises(CLIConfigError) as exc_info:
            resolve_config(
                command_name="labgym infer",
                defaults=INFER_DEFAULTS,
                allowed_keys=ALLOWED_INFER_KEYS,
                required_keys=REQUIRED_INFER_KEYS,
                config_path=str(cfg),
                cli_args_dict={}
            )
        assert f"Unknown configuration key(s) in '{cfg}' for command 'labgym infer': {forbidden_key}" in str(exc_info.value)


def test_load_categorizer_metadata_success(tmp_path):
    cat_dir = tmp_path / "my_categorizer"
    cat_dir.mkdir()
    param_file = cat_dir / "model_parameters.txt"
    with open(param_file, "w") as f:
        f.write("classnames,dim_conv,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\n")
        f.write("groom,64,1,15,0,0,0,0,0,0,0,1\n")
        f.write("walk,64,1,15,0,0,0,0,0,0,0,1\n")

    meta = load_categorizer_metadata(str(cat_dir))
    assert meta['classnames'] == ['groom', 'walk']
    assert meta['dim_conv'] == 64
    assert meta['channel'] == 1
    assert meta['length'] == 15
    assert meta['animation_analyzer'] is False
    assert meta['include_bodyparts'] is True
    assert meta['background_free'] is True
    assert meta['black_background'] is True
    assert meta['behavior_mode'] == 0
    assert 'groom' in meta['behaviornames_and_colors']
    assert 'walk' in meta['behaviornames_and_colors']


def test_load_categorizer_metadata_missing_and_malformed(tmp_path):
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(tmp_path / "nonexistent"))
    assert "Categorizer model directory not found" in str(exc_info.value)

    empty_dir = tmp_path / "empty_cat"
    empty_dir.mkdir()
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(empty_dir))
    assert "parameters file not found" in str(exc_info.value)

    bad_param = empty_dir / "model_parameters.txt"
    bad_param.write_text("invalid_column\n1\n")
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(empty_dir))
    assert "missing required field 'classnames'" in str(exc_info.value)

    # Test missing channel
    missing_channel = empty_dir / "model_parameters.txt"
    missing_channel.write_text("classnames,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance\ngroom,15,0,0,0,0,0,0,0\n")
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(empty_dir))
    assert "missing required field 'channel'" in str(exc_info.value)

    # Test missing dim_conv for network 0
    missing_dim = empty_dir / "model_parameters.txt"
    missing_dim.write_text("classnames,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance\ngroom,1,15,0,0,0,0,0,0,0\n")
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(empty_dir))
    assert "missing required architecture field 'dim_conv'" in str(exc_info.value)

    # Test missing dim_tconv for network 1
    missing_tconv = empty_dir / "model_parameters.txt"
    missing_tconv.write_text("classnames,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\ngroom,1,15,1,0,0,0,0,0,0,0\n")
    with pytest.raises(CLIConfigError) as exc_info:
        load_categorizer_metadata(str(empty_dir))
    assert "missing required architecture field 'dim_tconv'" in str(exc_info.value)


def test_load_detector_metadata(tmp_path):
    det_dir = tmp_path / "my_detector"
    det_dir.mkdir()
    param_file = det_dir / "model_parameters.txt"
    param_file.write_text(json.dumps({"animal_mapping": {"0": "mouse"}, "inferencing_framesize": 480}))

    meta = load_detector_metadata(str(det_dir))
    assert meta['animal_mapping'] == {"0": "mouse"}
    assert meta['inferencing_framesize'] == 480

    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(tmp_path / "nonexistent"))
    assert "Detector model directory not found" in str(exc_info.value)

    bad_det = tmp_path / "bad_det"
    bad_det.mkdir()
    (bad_det / "model_parameters.txt").write_text("not json")
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "Error reading Detector model parameters" in str(exc_info.value)

    # Test missing animal_mapping
    (bad_det / "model_parameters.txt").write_text(json.dumps({"inferencing_framesize": 480}))
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "missing required field 'animal_mapping'" in str(exc_info.value)

    # Test empty animal_mapping
    (bad_det / "model_parameters.txt").write_text(json.dumps({"animal_mapping": {}, "inferencing_framesize": 480}))
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "'animal_mapping' must be a non-empty dictionary" in str(exc_info.value)

    # Test missing inferencing_framesize
    (bad_det / "model_parameters.txt").write_text(json.dumps({"animal_mapping": {"0": "mouse"}}))
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "missing required field 'inferencing_framesize'" in str(exc_info.value)

    # Test invalid inferencing_framesize
    (bad_det / "model_parameters.txt").write_text(json.dumps({"animal_mapping": {"0": "mouse"}, "inferencing_framesize": -5}))
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "'inferencing_framesize' must be a positive integer" in str(exc_info.value)

    # Test boolean inferencing_framesize (JSON true)
    (bad_det / "model_parameters.txt").write_text('{"animal_mapping": {"0": "mouse"}, "inferencing_framesize": true}')
    with pytest.raises(CLIConfigError) as exc_info:
        load_detector_metadata(str(bad_det))
    assert "'inferencing_framesize' cannot be a boolean" in str(exc_info.value)


def test_resolve_media_paths(tmp_path):
    v1 = tmp_path / "v1.mp4"
    v1.write_text("video")
    v2 = tmp_path / "v2.avi"
    v2.write_text("video")
    txt = tmp_path / "notes.txt"
    txt.write_text("text")

    # Directory ignores non-media files
    resolved = resolve_media_paths([str(tmp_path)], behavior_mode=0)
    assert resolved == [str(v1), str(v2)]

    # Single media file resolves
    single = resolve_media_paths([str(v1)], behavior_mode=0)
    assert single == [str(v1)]

    # Deduplication preserves order
    dup = resolve_media_paths([str(v1), str(v2), str(v1)], behavior_mode=0)
    assert dup == [str(v1), str(v2)]

    # Reject unsupported individual files (e.g. notes.txt)
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_media_paths([str(txt)], behavior_mode=0)
    assert "Unsupported video file format" in str(exc_info.value)
    assert "notes.txt" in str(exc_info.value)

    # Nonexistent path
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_media_paths([str(tmp_path / "ghost.mp4")], behavior_mode=0)
    assert "Specified media path does not exist" in str(exc_info.value)


def test_handle_infer_missing_detector_when_use_detector_true(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_text("video")
    out = tmp_path / "never_created"
    exit_code = main(["infer", "--path-to-videos", str(v), "--result-path", str(out), "--use-detector"])
    assert exit_code == 2
    assert not out.exists()


def test_handle_infer_does_not_create_output_dir_on_failure(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("notes")
    out = tmp_path / "out_uncreated"

    # Fails because notes.txt is not a supported media file
    exit_code = main(["infer", "--path-to-videos", str(txt), "--result-path", str(out)])
    assert exit_code == 2
    assert not out.exists()


def test_handle_infer_unknown_behavior_requested(tmp_path):
    cat_dir = tmp_path / "cat"
    cat_dir.mkdir()
    (cat_dir / "model_parameters.txt").write_text("classnames,dim_conv,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\ngroom,64,1,15,0,0,0,0,0,0,0,1\n")
    v = tmp_path / "v.mp4"
    v.write_text("video")
    out = tmp_path / "out_uncreated_behavior"

    exit_code = main(["infer", "--path-to-videos", str(v), "--result-path", str(out), "--path-to-categorizer", str(cat_dir), "--behavior-to-include", "flying"])
    assert exit_code == 2
    assert not out.exists()


def test_handle_infer_e2e_animal_execution(tmp_path, capsys):
    cat_dir = tmp_path / "cat"
    cat_dir.mkdir()
    (cat_dir / "model_parameters.txt").write_text("classnames,dim_conv,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\ngroom,64,1,15,0,0,0,0,0,0,0,1\n")
    v = tmp_path / "v_n1.mp4"
    v.write_text("video")
    out = tmp_path / "out"

    class FakeAnimal:
        def __init__(self):
            self.event_probability = {0: [['groom', 0.95]]}
            self.all_time = [0.1]
        def prepare_analysis(self, *a, **k): pass
        def acquire_information(self, *a, **k): pass
        def craft_data(self): pass
        def categorize_behaviors(self, *a, **k): pass
        def annotate_video(self, *a, **k): pass
        def export_results(self, *a, **k): pass

    with patch('LabGym.analyzebehavior.AnalyzeAnimal', FakeAnimal), patch('LabGym.tools.plot_events'):
        exit_code = main(["infer", "--path-to-videos", str(v), "--result-path", str(out), "--path-to-categorizer", str(cat_dir)])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Inference completed successfully" in captured.out


def test_handle_infer_e2e_detector_execution(tmp_path, capsys):
    cat_dir = tmp_path / "cat"
    cat_dir.mkdir()
    (cat_dir / "model_parameters.txt").write_text("classnames,dim_conv,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\ngroom,64,1,15,0,0,0,0,0,0,0,1\n")
    det_dir = tmp_path / "det"
    det_dir.mkdir()
    (det_dir / "model_parameters.txt").write_text(json.dumps({"animal_mapping": {"0": "mouse"}, "inferencing_framesize": 480}))
    v = tmp_path / "v_n1.mp4"
    v.write_text("video")
    out = tmp_path / "out"

    class FakeAnimalDetector:
        def __init__(self):
            self.event_probability = {'mouse': {0: [['groom', 0.95]]}}
            self.all_time = [0.1]
        def prepare_analysis(self, *a, **k): pass
        def acquire_information(self, *a, **k): pass
        def craft_data(self): pass
        def categorize_behaviors(self, *a, **k): pass
        def annotate_video(self, *a, **k): pass
        def export_results(self, *a, **k): pass

    with patch('LabGym.analyzebehavior_dt.AnalyzeAnimalDetector', FakeAnimalDetector), patch('LabGym.tools.plot_events'):
        exit_code = main([
            "infer",
            "--path-to-videos", str(v),
            "--result-path", str(out),
            "--use-detector",
            "--path-to-detector", str(det_dir),
            "--path-to-categorizer", str(cat_dir)
        ])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Inference completed successfully" in captured.out


def test_path_to_videos_rejects_empty_strings():
    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['vid.mp4', ''], 'result_path': 'out/'}
        )
    assert "Items in 'path_to_videos' must be non-empty string paths" in str(exc_info.value)

    with pytest.raises(CLIConfigError) as exc_info:
        resolve_config(
            command_name="labgym infer",
            defaults=INFER_DEFAULTS,
            allowed_keys=ALLOWED_INFER_KEYS,
            required_keys=REQUIRED_INFER_KEYS,
            config_path=None,
            cli_args_dict={'path_to_videos': ['   '], 'result_path': 'out/'}
        )
    assert "Items in 'path_to_videos' must be non-empty string paths" in str(exc_info.value)


def test_handle_test_categorizer_validation_and_execution(tmp_path, capsys):
    gt_dir = tmp_path / "groundtruth"
    gt_dir.mkdir()
    cat_dir = tmp_path / "cat_model"
    cat_dir.mkdir()
    param_file = cat_dir / "model_parameters.txt"
    param_file.write_text("classnames,dim_conv,channel,time_step,network,inner_code,std,background_free,black_background,behavior_kind,social_distance,color_code\ngroom,64,1,15,0,0,0,0,0,0,0,1\n")
    out_dir = tmp_path / "test_cat_out"

    # Missing groundtruth_path
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(tmp_path / "nonexistent"), "--model-path", str(cat_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Groundtruth_path is a file
    dummy_file = tmp_path / "dummy_gt.txt"
    dummy_file.write_text("file")
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(dummy_file), "--model-path", str(cat_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Missing model_path
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(gt_dir), "--model-path", str(tmp_path / "nonexistent_model"), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Model_path missing model_parameters.txt
    empty_model = tmp_path / "empty_model"
    empty_model.mkdir()
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(gt_dir), "--model-path", str(empty_model), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # model_parameters.txt missing required field (e.g. time_step)
    bad_param_dir = tmp_path / "bad_param_model"
    bad_param_dir.mkdir()
    (bad_param_dir / "model_parameters.txt").write_text("classnames,network,inner_code,background_free\ngroom,0,0,0\n")
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(gt_dir), "--model-path", str(bad_param_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # model_parameters.txt missing classnames
    missing_classes_dir = tmp_path / "missing_classes_model"
    missing_classes_dir.mkdir()
    (missing_classes_dir / "model_parameters.txt").write_text("time_step,network,inner_code,background_free\n15,0,0,0\n")
    exit_code = main(["test", "categorizer", "--groundtruth-path", str(gt_dir), "--model-path", str(missing_classes_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Successful run
    with patch('LabGym.categorizer.Categorizers.test_categorizer') as mock_test:
        exit_code = main(["test", "categorizer", "--groundtruth-path", str(gt_dir), "--model-path", str(cat_dir), "--result-path", str(out_dir)])
        assert exit_code == 0
        mock_test.assert_called_once_with(str(gt_dir), str(cat_dir), result_path=str(out_dir))
        assert out_dir.exists()
        captured = capsys.readouterr()
        assert "Categorizer testing completed successfully." in captured.out


def test_handle_test_detector_validation_and_execution(tmp_path, capsys):
    anno_file = tmp_path / "annotations.json"
    anno_file.write_text("{}")
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    det_dir = tmp_path / "det_model"
    det_dir.mkdir()
    (det_dir / "model_parameters.txt").write_text('{"animal_mapping": {"0": "mouse"}, "inferencing_framesize": 480}')
    (det_dir / "config.yaml").write_text("MODEL:\n  DEVICE: cpu\n")
    (det_dir / "model_final.pth").write_text("weights")
    out_dir = tmp_path / "test_det_out"

    # Missing required result-path
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(det_dir)])
    assert exit_code == 2

    # Missing path_to_annotation
    exit_code = main(["test", "detector", "--path-to-annotation", str(tmp_path / "ghost.json"), "--path-to-testingimages", str(img_dir), "--model-path", str(det_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Annotation path is directory
    exit_code = main(["test", "detector", "--path-to-annotation", str(img_dir), "--path-to-testingimages", str(img_dir), "--model-path", str(det_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Missing path_to_testingimages
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(tmp_path / "ghost_imgs"), "--model-path", str(det_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Missing model_path
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(tmp_path / "ghost_model"), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Detector model directory missing config.yaml
    missing_config_dir = tmp_path / "missing_config_det"
    missing_config_dir.mkdir()
    (missing_config_dir / "model_parameters.txt").write_text("{}")
    (missing_config_dir / "model_final.pth").write_text("weights")
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(missing_config_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Detector model directory missing model_final.pth
    missing_pth_dir = tmp_path / "missing_pth_det"
    missing_pth_dir.mkdir()
    (missing_pth_dir / "model_parameters.txt").write_text("{}")
    (missing_pth_dir / "config.yaml").write_text("cfg")
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(missing_pth_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Detector model directory missing model_parameters.txt
    missing_param_dir = tmp_path / "missing_param_det"
    missing_param_dir.mkdir()
    (missing_param_dir / "config.yaml").write_text("cfg")
    (missing_param_dir / "model_final.pth").write_text("weights")
    exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(missing_param_dir), "--result-path", str(out_dir)])
    assert exit_code == 2
    assert not out_dir.exists()

    # Successful run
    with patch('LabGym.detector.Detector.test') as mock_test:
        exit_code = main(["test", "detector", "--path-to-annotation", str(anno_file), "--path-to-testingimages", str(img_dir), "--model-path", str(det_dir), "--result-path", str(out_dir)])
        assert exit_code == 0
        mock_test.assert_called_once_with(str(anno_file), str(img_dir), str(det_dir), str(out_dir))
        assert out_dir.exists()
        captured = capsys.readouterr()
        assert "Detector testing completed successfully." in captured.out


def test_handle_test_detector_unreadable_file_fails_without_creating_out_dir(tmp_path):
    anno_file = tmp_path / "annotations.json"
    anno_file.write_text("{}")
    img_dir = tmp_path / "test_images"
    img_dir.mkdir()
    det_dir = tmp_path / "det_model"
    det_dir.mkdir()
    (det_dir / "model_parameters.txt").write_text('{"animal_mapping": {"0": "mouse"}, "inferencing_framesize": 480}')
    (det_dir / "config.yaml").write_text("MODEL:\n  DEVICE: cpu\n")
    (det_dir / "model_final.pth").write_text("weights")
    out_dir = tmp_path / "test_det_out_unreadable"

    orig_open = open

    # Simulate read failure on model_final.pth (binary read)
    def mock_open_weights(file, *args, **kwargs):
        if str(file).endswith("model_final.pth"):
            raise PermissionError("Permission denied: unreadable weights")
        return orig_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_weights):
        exit_code = main([
            "test", "detector",
            "--path-to-annotation", str(anno_file),
            "--path-to-testingimages", str(img_dir),
            "--model-path", str(det_dir),
            "--result-path", str(out_dir)
        ])
        assert exit_code == 2
        assert not out_dir.exists()

    # Simulate read failure on config.yaml (text read)
    def mock_open_config(file, *args, **kwargs):
        if str(file).endswith("config.yaml"):
            raise OSError("I/O error reading config")
        return orig_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_config):
        exit_code = main([
            "test", "detector",
            "--path-to-annotation", str(anno_file),
            "--path-to-testingimages", str(img_dir),
            "--model-path", str(det_dir),
            "--result-path", str(out_dir)
        ])
        assert exit_code == 2
        assert not out_dir.exists()

    # Simulate read failure on model_parameters.txt (text read)
    def mock_open_params(file, *args, **kwargs):
        if str(file).endswith("model_parameters.txt"):
            raise PermissionError("Permission denied: unreadable model parameters")
        return orig_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_params):
        exit_code = main([
            "test", "detector",
            "--path-to-annotation", str(anno_file),
            "--path-to-testingimages", str(img_dir),
            "--model-path", str(det_dir),
            "--result-path", str(out_dir)
        ])
        assert exit_code == 2
        assert not out_dir.exists()


def test_handle_train_categorizer_validation_and_execution(tmp_path, capsys):
    data_dir = tmp_path / "behavior_data"
    data_dir.mkdir()
    model_dir = tmp_path / "cat_model"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("file")

    # Missing data_path
    exit_code = main(["train", "categorizer", "--data-path", str(tmp_path / "nonexistent_data"), "--model-path", str(model_dir)])
    assert exit_code == 2
    assert not model_dir.exists()

    # data_path is a file, not a directory
    exit_code = main(["train", "categorizer", "--data-path", str(dummy_file), "--model-path", str(model_dir)])
    assert exit_code == 2
    assert not model_dir.exists()

    # model_path is an existing file
    exit_code = main(["train", "categorizer", "--data-path", str(data_dir), "--model-path", str(dummy_file)])
    assert exit_code == 2

    # out_path is an existing file
    exit_code = main(["train", "categorizer", "--data-path", str(data_dir), "--model-path", str(model_dir), "--out-path", str(dummy_file)])
    assert exit_code == 2
    assert not model_dir.exists()

    # out_folder is an existing file
    exit_code = main(["train", "categorizer", "--data-path", str(data_dir), "--model-path", str(model_dir), "--out-folder", str(dummy_file)])
    assert exit_code == 2
    assert not model_dir.exists()

    # Invalid aug_methods
    exit_code = main(["train", "categorizer", "--data-path", str(data_dir), "--model-path", str(model_dir), "--aug-methods", "bad_aug"])
    assert exit_code == 2
    assert not model_dir.exists()

    # Successful combnet training
    out_path = tmp_path / "reports"
    with patch('LabGym.categorizer.Categorizers.train_combnet') as mock_train:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--out-path", str(out_path),
            "--network-type", "combnet",
            "--aug-methods", "rotation", "vertical flipping"
        ])
        assert exit_code == 0
        assert model_dir.exists()
        assert out_path.exists()
        mock_train.assert_called_once()
        _, kwargs = mock_train.call_args
        assert kwargs['aug_methods'] == ['random rotation', 'vertical flipping']
        captured = capsys.readouterr()
        assert "Categorizer training completed successfully." in captured.out


def test_handle_train_categorizer_network_dispatch(tmp_path):
    data_dir = tmp_path / "behavior_data"
    data_dir.mkdir()
    model_dir = tmp_path / "cat_model"

    # Pattern recognizer
    with patch('LabGym.categorizer.Categorizers.train_pattern_recognizer') as mock_pr:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--network-type", "pattern_recognizer"
        ])
        assert exit_code == 0
        mock_pr.assert_called_once()

    # Pattern recognizer on-the-fly
    with patch('LabGym.categorizer.Categorizers.train_pattern_recognizer_onfly') as mock_pr_onfly:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--network-type", "pattern_recognizer",
            "--training-onfly"
        ])
        assert exit_code == 0
        mock_pr_onfly.assert_called_once()

    # Animation analyzer
    with patch('LabGym.categorizer.Categorizers.train_animation_analyzer') as mock_aa:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--network-type", "animation_analyzer"
        ])
        assert exit_code == 0
        mock_aa.assert_called_once()

    # Animation analyzer on-the-fly
    with patch('LabGym.categorizer.Categorizers.train_animation_analyzer_onfly') as mock_aa_onfly:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--network-type", "animation_analyzer",
            "--training-onfly"
        ])
        assert exit_code == 0
        mock_aa_onfly.assert_called_once()

    # Combnet on-the-fly
    with patch('LabGym.categorizer.Categorizers.train_combnet_onfly') as mock_cb_onfly:
        exit_code = main([
            "train", "categorizer",
            "--data-path", str(data_dir),
            "--model-path", str(model_dir),
            "--network-type", "combnet",
            "--training-onfly"
        ])
        assert exit_code == 0
        mock_cb_onfly.assert_called_once()


def test_handle_train_detector_validation_and_execution(tmp_path, capsys):
    anno_file = tmp_path / "annotations.json"
    anno_file.write_text(json.dumps({"categories": [{"id": 1, "name": "mouse"}]}))
    img_dir = tmp_path / "train_images"
    img_dir.mkdir()
    det_dir = tmp_path / "det_model"
    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("dummy")

    # Missing path_to_annotation
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(tmp_path / "nonexistent.json"),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # path_to_annotation is directory
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(img_dir),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # path_to_annotation is invalid JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("invalid json")
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(bad_json),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # path_to_annotation missing 'categories' list
    no_cat_json = tmp_path / "no_cat.json"
    no_cat_json.write_text(json.dumps({"images": []}))
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(no_cat_json),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # Missing path_to_trainingimages
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(anno_file),
        "--path-to-trainingimages", str(tmp_path / "nonexistent_imgs"),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # path_to_trainingimages is a file
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(anno_file),
        "--path-to-trainingimages", str(dummy_file),
        "--model-path", str(det_dir)
    ])
    assert exit_code == 2
    assert not det_dir.exists()

    # model_path is an existing file
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(anno_file),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(dummy_file)
    ])
    assert exit_code == 2

    # model_path already exists as a directory
    existing_dir = tmp_path / "existing_det"
    existing_dir.mkdir()
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(anno_file),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(existing_dir)
    ])
    assert exit_code == 2

    # Parent directory of model_path is a file
    file_parent = tmp_path / "file_parent"
    file_parent.write_text("file")
    exit_code = main([
        "train", "detector",
        "--path-to-annotation", str(anno_file),
        "--path-to-trainingimages", str(img_dir),
        "--model-path", str(file_parent / "det_model")
    ])
    assert exit_code == 2

    # Successful detector training run
    with patch('LabGym.detector.Detector.train') as mock_train:
        exit_code = main([
            "train", "detector",
            "--path-to-annotation", str(anno_file),
            "--path-to-trainingimages", str(img_dir),
            "--model-path", str(det_dir),
            "--iteration-num", "500",
            "--inference-size", "600"
        ])
        assert exit_code == 0
        mock_train.assert_called_once_with(
            str(anno_file), str(img_dir), str(det_dir), 500, 600
        )
        captured = capsys.readouterr()
        assert "Detector training completed successfully." in captured.out


def test_train_rejects_irrelevant_cross_command_parameters(tmp_path):
    # Reject detector-only params in categorizer training
    cfg_cat = tmp_path / "cat_bad.yaml"
    with open(cfg_cat, "w") as f:
        yaml.dump({
            'data_path': str(tmp_path),
            'model_path': str(tmp_path / 'model'),
            'path_to_annotation': 'anno.json'
        }, f)
    exit_code = main(["train", "categorizer", "-c", str(cfg_cat)])
    assert exit_code == 2

    # Reject categorizer-only params in detector training
    cfg_det = tmp_path / "det_bad.yaml"
    with open(cfg_det, "w") as f:
        yaml.dump({
            'path_to_annotation': 'anno.json',
            'path_to_trainingimages': str(tmp_path),
            'model_path': str(tmp_path / 'model'),
            'data_path': str(tmp_path)
        }, f)
    exit_code = main(["train", "detector", "-c", str(cfg_det)])
    assert exit_code == 2
