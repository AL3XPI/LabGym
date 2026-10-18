import os
import sys
import pytest
import yaml

from LabGym.cli import (
    build_parser,
    resolve_config,
    CLIConfigError,
    main,
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
