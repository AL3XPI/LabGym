"""Compare incremental Detector analysis with the full-recording path."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import cv2
import numpy as np

from LabGym import analyzebehavior_dt as analysis


class ArrayOutput:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class SyntheticVideo:
    frame_count = 75

    def __init__(self, path):
        self.index = 0

    def get(self, property_id):
        return 25 if property_id == cv2.CAP_PROP_FPS else self.frame_count

    def read(self):
        if self.index == self.frame_count:
            return False, None
        frame = np.full((64, 64, 3), self.index, dtype=np.uint8)
        frame[:, 32:] += 80
        self.index += 1
        return True, frame

    def release(self):
        pass


class SyntheticDetector:
    animal_mapping = {'0': 'animal'}

    def load(self, path, kinds):
        pass

    def inference(self, inputs):
        frame = int(inputs[0]['image'][0, 0, 0])
        masks = []
        # One late arrival, a brief gap, and two spatially separated IDs.
        for track, start in enumerate((0, 9)):
            if frame < start or (track == 0 and frame in (22, 23, 24)):
                continue
            x = 8 + track * 34 + frame % 5
            mask = np.zeros((64, 64), dtype=bool)
            mask[12:29, x:x + 13] = True
            masks.append(mask)
        instances = SimpleNamespace(
            pred_masks=ArrayOutput(np.asarray(masks).reshape(-1, 64, 64)),
            pred_classes=ArrayOutput(np.zeros(len(masks), dtype=int)),
            scores=ArrayOutput(np.ones(len(masks))),
        )
        instances.to = lambda device: instances
        return [{'instances': instances}]


class RecordingCategorizer:
    def __init__(self, multiclass):
        self.batches = []
        self.multiclass = multiclass

    def predict(self, inputs, batch_size):
        arrays = inputs if isinstance(inputs, list) else [inputs]
        arrays = tuple(np.asarray(array) for array in arrays)
        self.batches.append(arrays)
        means = [array.mean(axis=tuple(range(1, array.ndim))) for array in arrays]
        score = (sum(means) * 13) % 1
        if self.multiclass:
            return np.column_stack((score * 0.8, (1 - score) * 0.8, np.full_like(score, 0.2)))
        return score[:, None]


class IncrementalCategorizerTest(TestCase):
    def run_analysis(self, output, model, *, incremental, animation, include_bodyparts):
        analyzer = analysis.AnalyzeAnimalDetector()
        colors = {'idle': ['white', 'black'], 'moving': ['red', 'white']}
        if model.multiclass:
            colors['grooming'] = ['black', 'green']
        analyzer.prepare_analysis(
            'detector', 'video.avi', output, {'animal': 2}, ['animal'], 0,
            names_and_colors=colors,
            length=4, duration=0, animation_analyzer=animation,
            include_bodyparts=include_bodyparts,
            incremental_categorizer='categorizer' if incremental else None,
            categorizer_uncertain=0.15,
        )
        high_water = []
        keys = []
        consume = analyzer._consume_windows
        export_probabilities = analyzer.export_probability_matrices_dt

        def observe_consumption():
            high_water.append(len(analyzer._pending_windows))
            keys.extend(window[:3] for window in analyzer._pending_windows)
            consume()
            self.assertFalse(analyzer._pending_windows)

        with patch.object(analyzer, '_consume_windows', side_effect=observe_consumption), \
                patch.object(analysis, 'load_model', return_value=model) as load, \
                patch.object(analyzer, 'export_probability_matrices_dt',
                             side_effect=lambda: export_probabilities(make_heatmap=False)) as export:
            analyzer.acquire_information(batch_size=1, background_free=False)
            if incremental:
                self.assertGreaterEqual(len(model.batches), 2)
                self.assertEqual(analyzer.pattern_images, {'animal': {}})
                self.assertEqual(analyzer.animations, {})
                self.assertFalse(analyzer._frame_patterns)
                self.assertTrue(all(size == 32 for size in high_water))
                self.assertGreater(len(analyzer._pending_windows), 0)
                self.assertLess(len(analyzer._pending_windows), 32)
            else:
                self.assertFalse(model.batches)
            analyzer.craft_data()
            analyzer.categorize_behaviors('categorizer', uncertain=0.15, min_length=3)
            export.assert_called_once_with()
            load.assert_called_once_with('categorizer')
        analyzer.analyze_parameters(parameter_to_analyze=['count', 'duration', 'latency'])
        return analyzer, keys

    def test_multiple_chunks_preserve_inputs_and_final_results(self):
        for animation, inner in ((True, True), (False, False)):
            with self.subTest(animation=animation), TemporaryDirectory() as output, \
                    patch.object(analysis, 'Detector', SyntheticDetector), \
                    patch.object(analysis.cv2, 'VideoCapture', SyntheticVideo):
                reference_model = RecordingCategorizer(multiclass=animation)
                actual_model = RecordingCategorizer(multiclass=animation)
                reference, _ = self.run_analysis(
                    Path(output) / 'reference', reference_model, incremental=False, animation=animation,
                    include_bodyparts=inner,
                )
                actual, keys = self.run_analysis(
                    Path(output) / 'incremental', actual_model, incremental=True, animation=animation,
                    include_bodyparts=inner,
                )

                times = len(reference.all_time)
                self.assertEqual(actual.all_time, reference.all_time)
                self.assertEqual(actual.register_counts, reference.register_counts)
                self.assertEqual(actual.animal_centers, reference.animal_centers)
                self.assertEqual(actual.event_probability, reference.event_probability)
                self.assertEqual(keys, sorted(set(keys), key=lambda key: (key[2], key[1])))
                expected = []
                for track in range(2):
                    for label, old_parameters in reference.all_behavior_parameters['animal'].items():
                        new_parameters = actual.all_behavior_parameters['animal'][label]
                        old = old_parameters['probability'][track]
                        np.testing.assert_array_equal(new_parameters['probability'][track], old)
                        for parameter in ('count', 'duration', 'latency'):
                            self.assertEqual(new_parameters[parameter][track], old_parameters[parameter][track])
                    expected.extend(('animal', track, frame) for frame in range(times) if not np.isnan(old[frame]))
                self.assertEqual(set(keys), set(expected))
                indexes = [track * times + frame for _, track, frame in keys]
                for legacy, *chunks in zip(reference_model.batches[0], *actual_model.batches):
                    np.testing.assert_array_equal(np.concatenate(chunks), legacy[indexes])
                for path in Path(reference.results_path).glob('*probability_matrix*'):
                    self.assertEqual((Path(actual.results_path) / path.name).read_bytes(), path.read_bytes())
