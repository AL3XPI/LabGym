import argparse
import sys
import pytest
import yaml
import numpy as np

from LabGym.cli import str2bool, merge_configs, get_centroid, parse_pair

def test_str2bool():
    # Test typical true values
    assert str2bool("true") is True
    assert str2bool("T") is True
    assert str2bool("1") is True
    assert str2bool("yes") is True
    assert str2bool(True) is True
    
    # Test typical false values
    assert str2bool("false") is False
    assert str2bool("F") is False
    assert str2bool("0") is False
    assert str2bool("no") is False
    assert str2bool(False) is False
    
    # Test error cases
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("not-a-bool")

def test_parse_pair():
    assert parse_pair("BehaviorA -> BehaviorB") == ("BehaviorA", "BehaviorB")
    assert parse_pair("BehaviorA -> BehaviorB (5 errors)") == ("BehaviorA", "BehaviorB")
    assert parse_pair("InvalidPair") is None

def test_merge_configs(tmp_path):
    # Setup mock CLI args
    class MockArgs:
        def __init__(self):
            self.config = None
            self.dim_conv = None
            self.channel = None
            self.background_free = None
            
    cli_args = MockArgs()
    cli_args.dim_conv = 128
    cli_args.background_free = True
    
    defaults = {
        'dim_conv': 64,
        'channel': 3,
        'background_free': False,
    }
    
    # Merge without config file
    merged = merge_configs(cli_args, defaults)
    assert merged['dim_conv'] == 128  # overridden by CLI
    assert merged['channel'] == 3     # fallback to default
    assert merged['background_free'] is True # overridden by CLI
    
    # Merge with YAML config file
    config_file = tmp_path / "config.yaml"
    config_data = {
        'dim_conv': 32,
        'channel': 1,
        'background_free': False,
    }
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
        
    cli_args.config = str(config_file)
    cli_args.dim_conv = None  # CLI does not override this one
    cli_args.background_free = True # CLI overrides this one
    
    merged_with_yaml = merge_configs(cli_args, defaults)
    assert merged_with_yaml['dim_conv'] == 32  # from YAML
    assert merged_with_yaml['channel'] == 1     # from YAML
    assert merged_with_yaml['background_free'] is True # from CLI overriding YAML

def test_get_centroid():
    embedding_map = {
        "file1.avi": np.array([1.0, 0.0, 0.0]),
        "file2.avi": np.array([0.9, 0.1, 0.0]),
        "file3.avi": np.array([0.0, 1.0, 0.0])
    }
    
    # Test centroid calculation
    file_list = ["file1.avi", "file2.avi", "file3.avi"]
    centroid = get_centroid(embedding_map, file_list)
    
    # Mean of embs is [1.9/3, 1.1/3, 0.0] = [0.633, 0.366, 0.0]
    # Cosine similarities to mean:
    # file1: dot([1,0,0], mean) = 0.633. Norm: 1. Sim: 0.633
    # file2: dot([0.9, 0.1, 0.0], mean) = 0.9*0.633 + 0.1*0.366 = 0.57 + 0.036 = 0.606. Norm: sqrt(0.81+0.01) = 0.905. Sim: 0.606/0.905 = 0.67
    # file3: dot([0,1,0], mean) = 0.366. Norm: 1. Sim: 0.366
    # file2.avi should be the centroid
    assert centroid == "file2.avi"
    
    # Test with single file
    assert get_centroid(embedding_map, ["file3.avi"]) == "file3.avi"
    
    # Test with empty file list
    assert get_centroid(embedding_map, []) is None

def test_merge_detector_configs():
    class MockArgs:
        def __init__(self):
            self.config = None
            self.path_to_annotation = None
            self.path_to_trainingimages = None
            self.iteration_num = 500
            self.inference_size = None
            
    cli_args = MockArgs()
    defaults = {
        'path_to_annotation': None,
        'path_to_trainingimages': None,
        'iteration_num': 200,
        'inference_size': 480,
    }
    
    merged = merge_configs(cli_args, defaults)
    assert merged['iteration_num'] == 500
    assert merged['inference_size'] == 480

