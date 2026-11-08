import unittest

import numpy as np

from analysis import BASELINE, CARDS, DATA, PREDICTIONS, Point, Sweep, load


class ThresholdTests(unittest.TestCase):
    def test_tied_scores_and_boundaries(self):
        sweep = Sweep([0, 1, 1, 0], [0, 0.2, 0.2, 0.9])
        self.assertEqual(sweep.at(0), Point(0, 2, 2, 0, 0))
        self.assertEqual(sweep.at(0.2), Point(0.2, 2, 1, 0, 1))
        self.assertEqual(sweep.at(1), Point(1, 0, 0, 2, 2))
        self.assertIsNone(sweep.at(1).precision)
        self.assertEqual(sweep.at(1).fbeta(4), 0)

    def test_optima_against_direct_counting(self):
        y = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0, .18, .18, .3, .5, .6])
        sweep = Sweep(y, scores)
        points = []
        for tau in [0, .18, .3, .5, .6, 1]:
            predicted = scores >= tau
            points.append(Point(tau, int((predicted & (y == 1)).sum()), int((predicted & (y == 0)).sum()),
                                int((~predicted & (y == 1)).sum()), int((~predicted & (y == 0)).sum())))
        for k in [1, 4, 10]:
            self.assertEqual(sweep.best(k), (min(points, key=lambda p: (p.cost(k), -p.threshold)),
                                             max(points, key=lambda p: (p.fbeta(k), p.threshold))))


@unittest.skipUnless(all((DATA / p).is_file() for p in [PREDICTIONS, CARDS]), "Скачайте данные курса")
class CourseLogTests(unittest.TestCase):
    def test_baseline_and_task(self):
        log = load()
        self.assertEqual((log.total, len(log.labeled), log.missing_label, log.excluded_label), (19393, 15839, 2950, 604))
        self.assertEqual(log.sweep.at(BASELINE), Point(.18, 1331, 2563, 335, 11610))
        cost, f = log.sweep.best(4)
        self.assertAlmostEqual(cost.threshold, .25267)
        self.assertAlmostEqual(f.threshold, .25129)
        self.assertEqual((cost.cost(4), f.cost(4)), (3364, 3365))
        self.assertLess(cost.fbeta(4), f.fbeta(4))
        self.assertEqual(len(log.error_examples(.18, "FN")), 10)
        self.assertTrue(log.error_examples(1, "FP").empty)


if __name__ == "__main__":
    unittest.main()
