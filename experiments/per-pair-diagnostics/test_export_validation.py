import unittest

from validate_exports import compare_interval, interval, validate_post_hoc


class ExportValidationTests(unittest.TestCase):
    def test_independent_interval_reconstruction_detects_wrong_se_or_critical_value(self):
        expected = interval([1, 2, 3, 4, 5])
        self.assertAlmostEqual(expected['se'], 1 / 2**.5)
        self.assertAlmostEqual(expected['ci_high'], 3 + 2.7764451051977987 / 2**.5)
        row = {key: str(value) for key, value in expected.items()}
        compare_interval(row, [1, 2, 3, 4, 5])
        row['se'] = '0.632455532'
        with self.assertRaisesRegex(ValueError, 'interval'):
            compare_interval(row, [1, 2, 3, 4, 5])

    def test_post_hoc_label_is_required_before_any_statistics_are_accepted(self):
        row = dict(post_hoc='False', view='uniform_pairs', nominal_step='100', maze='maze4')
        with self.assertRaisesRegex(ValueError, 'label'):
            validate_post_hoc([row], {}, dict(maze4=100), 'uniform_pairs')


if __name__ == '__main__':
    unittest.main()
