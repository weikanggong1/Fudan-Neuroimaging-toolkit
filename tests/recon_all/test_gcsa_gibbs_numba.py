"""Numba有序Gibbs与既有Python评分/种子/每轮标签合同。"""
from dataclasses import replace
import unittest
import numpy as np

from fnit.recon_all.gcsa_gibbs import GibbsModel
from fnit.recon_all.gcsa_gibbs_numba import pack_model, neighborhood_likelihood
from fnit.recon_all.gcsa_initial import InitialAtlas
from fnit.recon_all.gcsa_reclassify import reclassify_gibbs


def example_model(*, tied=False, duplicate_choices=False):
    nodes = [((11, 10, .15, .7), (22, 12, -.3, 1.4))]
    prior = [((11, .6), (22, .4))]
    if tied:
        nodes = [((11, 10, 0., 1.), (22, 12, 0., 1.))]
        prior = [((11, .5), (22, .5))]
    if duplicate_choices:
        prior = [((11, .6), (22, .4), (11, .6))]
    chances = ({11: .65, 22: .35}, {11: 0., 22: 1.})
    if tied: chances = ({11: .5, 22: .5}, {11: .5, 22: .5})
    atlas = InitialAtlas(classifier_nodes=nodes, prior_nodes=prior, color_table={},
        source_name="test", average_variance=1., minimum_determinant=.01,
        singular_count=0, regularized_count=0,
        gibbs_neighbours=[tuple(chances for _ in prior[0])])
    count = 7
    angle = np.arange(count) * (2 * np.pi / count)
    xyz = np.stack((np.cos(angle), np.sin(angle), angle * .1), axis=1).astype(np.float32)
    neighbors = [[(v + 1) % count, (v - 1) % count] for v in range(count)]
    directions = np.tile(np.eye(3, dtype=np.float32)[:2], (count, 1, 1))
    model = GibbsModel(atlas, np.zeros(count, np.int64), np.zeros(count, np.int64),
        np.linspace(-.9, .8, count, dtype=np.float32), xyz, neighbors, directions,
        np.array([22, 11, 22, 11, 22, 22, 11], np.int32))
    return model, atlas


class GibbsNumba(unittest.TestCase):
    def test_scores_and_temporary_label_restore(self):
        model, atlas = example_model()
        packed = pack_model(model, atlas)
        old = model.labels.copy()
        for vertex in range(len(old)):
            for label in (11, 22, 999):
                value = float(model.feature[vertex])
                expected = model.neighborhood_log_likelihood(vertex, label)
                actual = neighborhood_likelihood(packed=packed, vertex=vertex,
                    candidate=label, input_value=value, labels=model.labels)
                self.assertEqual(actual, expected)
                np.testing.assert_array_equal(model.labels, old)

    def test_seeded_each_iteration_matches(self):
        for duplicate in (False, True):
            with self.subTest(duplicate_choices=duplicate):
                old, atlas = example_model(duplicate_choices=duplicate)
                new, _ = example_model(duplicate_choices=duplicate)
                old_steps, new_steps = [], []
                a = reclassify_gibbs(old, atlas, seed=1234, backend="python",
                    snapshot=lambda iteration, labels: old_steps.append(labels.copy()))
                b = reclassify_gibbs(new, atlas, seed=1234, backend="numba",
                    snapshot=lambda iteration, labels: new_steps.append(labels.copy()))
                self.assertEqual(a, b)
                self.assertEqual(len(old_steps), len(new_steps))
                for old_labels, new_labels in zip(old_steps, new_steps):
                    np.testing.assert_array_equal(old_labels, new_labels)

    def test_equal_score_keeps_current_label(self):
        model, atlas = example_model(tied=True)
        expected = model.labels.copy()
        history = reclassify_gibbs(model, atlas, seed=1234, backend="numba")
        np.testing.assert_array_equal(model.labels, expected)
        self.assertEqual(history, [{"iteration": 0, "changed": 0, "examined": len(expected)}])

    def test_invalid_backend_fails(self):
        model, atlas = example_model()
        with self.assertRaises(ValueError):
            reclassify_gibbs(model, atlas, backend="parallel")

    def test_invalid_variance_rejected_in_candidate(self):
        model, atlas = example_model()
        model.classifier[0][11] = (0., 0.)
        with self.assertRaisesRegex(ValueError, "positive classifier variance"):
            pack_model(model, atlas)


if __name__ == "__main__":
    unittest.main()
