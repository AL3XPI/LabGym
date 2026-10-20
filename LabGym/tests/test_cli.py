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
