"""Detector reuse is scoped to a run and independent of per-video analysis."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from LabGym import analyzebehavior_dt as analysis
from LabGym import detector as detector_module
from LabGym.detector import Detector, DetectorReuse


class VideoHeader:
    def __init__(self, path):
        self.path = path

    def get(self, property_id):
        return 25 if property_id == cv2.CAP_PROP_FPS else 120

    def read(self):
        value = 1 if self.path == 'first.avi' else 2
        return True, np.full((64, 96, 3), value, dtype=np.uint8)

    def release(self):
        pass


def write_artifact(path):
    path.mkdir()
    (path / 'config.yaml').write_text('MODEL:\n  ROI_HEADS:\n    SCORE_THRESH_TEST: 0.5\n')
    (path / 'model_final.pth').write_bytes(b'test weights')
    (path / 'model_parameters.txt').write_text('{}')
    return path


def load_detector(detector, path, kinds, *, config=None):
    detector.animal_mapping = {'0': 'animal'}
    detector.current_detector = object()


class DetectorReuseTest(TestCase):
    def test_consecutive_videos_share_only_detector_model(self):
        with TemporaryDirectory() as directory, \
                patch.object(Detector, 'load', autospec=True, side_effect=load_detector) as load, \
                patch.object(analysis.cv2, 'VideoCapture', VideoHeader):
            artifact = write_artifact(Path(directory) / 'detector')
            reuse = DetectorReuse()
            first = analysis.AnalyzeAnimalDetector()
            second = analysis.AnalyzeAnimalDetector()
            for analyzer, path in ((first, 'first.avi'), (second, 'second.avi')):
                analyzer.prepare_analysis(
                    artifact, path, directory, {'animal': 1}, ['animal'], 0,
                    names_and_colors={'walking': ['white', 'black']}, detector_reuse=reuse,
                )
                if analyzer is first:
                    analyzer.animal_centers['animal'][0][0] = (20, 30)
                    analyzer.skipped_frames.append(5)

            self.assertEqual(load.call_count, 1)
            self.assertIs(first.detector, second.detector)
            self.assertEqual(second.animal_mapping, {'0': 'animal'})
            self.assertEqual(first.animal_centers['animal'][0][0], (20, 30))
            self.assertIsNone(second.animal_centers['animal'][0][0])
            self.assertEqual(second.skipped_frames, [])
            self.assertNotEqual(first.results_path, second.results_path)
            self.assertFalse(np.array_equal(first.background, second.background))
            self.assertIsNot(DetectorReuse().load(artifact, ['animal']), first.detector)

    def test_configuration_weights_and_path_changes_reload(self):
        with TemporaryDirectory() as directory, \
                patch.object(Detector, 'load', autospec=True, side_effect=load_detector) as load:
            path = write_artifact(Path(directory) / 'first')
            another_path = write_artifact(Path(directory) / 'second')
            config = path / 'config.yaml'
            base = path / 'base.yaml'
            base.write_text(config.read_text())
            config.write_text('_BASE_: base.yaml\n')
            reuse = DetectorReuse()
            first = reuse.load(path, ['animal'])
            self.assertIs(first, reuse.load(path, ['animal']))
            base.write_text(base.read_text().replace('0.5', '0.6'))
            different_config = reuse.load(path, ['animal'])
            self.assertIsNot(different_config, first)
            self.assertEqual(load.call_args.kwargs['config'].MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.6)
            (path / 'model_final.pth').write_bytes(b'new test weights')
            changed_weights = reuse.load(path, ['animal'])
            self.assertIsNot(changed_weights, different_config)
            self.assertIsNot(reuse.load(another_path, ['animal']), changed_weights)
            self.assertEqual(load.call_count, 4)

    def test_device_change_reloads_detector(self):
        with TemporaryDirectory() as directory, \
                patch.object(Detector, 'load', autospec=True, side_effect=load_detector) as load, \
                patch.object(detector_module.torch.cuda, 'is_available', return_value=False) as cuda_available:
            path = write_artifact(Path(directory) / 'detector')
            reuse = DetectorReuse()
            first = reuse.load(path, ['animal'])
            self.assertEqual(first.device, 'cpu')
            self.assertEqual(load.call_args.kwargs['config'].MODEL.DEVICE, 'cpu')
            self.assertIs(reuse.load(path, ['animal']), first)
            self.assertEqual(load.call_count, 1)

            cuda_available.return_value = True
            second = reuse.load(path, ['animal'])
            self.assertIsNot(second, first)
            self.assertIsNot(second.current_detector, first.current_detector)
            self.assertEqual(second.device, 'cuda')
            self.assertIs(load.call_args.args[0], second)
            self.assertEqual(load.call_args.kwargs['config'].MODEL.DEVICE, 'cuda')
            self.assertEqual(load.call_count, 2)
