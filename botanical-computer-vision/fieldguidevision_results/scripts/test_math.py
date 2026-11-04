"""CPU checks of known geometry and evaluation semantics; no models needed."""
import unittest
import numpy as np
from run import audit_target, build_targets, unit


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([[3, 0], [1, 0], [0, 4], [0, 2], [-2, 0], [0, -2], [-1, -1], [1, 1]], dtype=np.float32)
        self.groups = {5: np.array([0, 0, 1, 1, 2, 3, 4, 4]), 7: np.array([0, 0, 1, 2, 3, 4, 5, 6])}
        self.organs = np.array(["leaf", "leaf", "fruit", "bark", "bark", "bark", "bark", "bark"])

    def test_raw_mean_and_linear_walk(self):
        targets = build_targets(self.raw, self.groups, self.organs)
        self.assertEqual(len(targets), 17)
        np.testing.assert_allclose(targets[5]["embedding"], [2, 0])
        np.testing.assert_allclose(targets[14]["embedding"], [1, 2])
        self.assertIsNone(targets[14]["group"])
        self.assertEqual(targets[12]["group"], 0)
        self.assertEqual(targets[16]["group"], 1)

    def test_cosine_retrieval_and_hit_denominator(self):
        t = {"id": "a", "k": 7, "group": 0, "embedding": np.array([8, 0])}
        gen = np.array([[1, 0], [2, 0], [0, 1], [-1, 0]])
        result = audit_target(t, gen, self.raw, self.groups[7], self.organs, np.array(list("abcdefgh")))
        self.assertEqual(result["target_group_hits"], 2)
        self.assertEqual(result["target_group_share"], .5)
        self.assertEqual(result["samples"][0]["nearest_image_id"], "a")

    def test_missing_endpoint_fails(self):
        with self.assertRaisesRegex(ValueError, "fruit"):
            build_targets(self.raw, self.groups, np.array(["leaf"]*8))

    def test_explicit_fruit_fallback_keeps_actual_majority(self):
        organs = np.array(["leaf", "leaf", "fruit", "twig", "bark", "bark", "bark", "bark"])
        groups = {5: self.groups[5], 7: np.array([0, 1, 2, 2, 3, 4, 5, 6])}
        # A lexical tie makes fruit the majority here, so add a third twig to
        # force a true mixed endpoint, while retaining seven unique groups.
        raw = np.concatenate([self.raw, [[0, 3]]])
        groups = {5: np.r_[groups[5], 1], 7: np.r_[groups[7], 2]}
        organs = np.concatenate([organs, ["twig"]])
        targets = build_targets(raw, groups, organs, fruit_fallback=True)
        self.assertEqual(targets[-1]["fruit_endpoint_majority_organ"], "twig")
        self.assertAlmostEqual(targets[-1]["fruit_endpoint_fruit_share"], 1/3)

    def test_invalid_vectors_rejected(self):
        for x in ([0, 0], [float("nan"), 1]):
            with self.assertRaises(ValueError):
                unit(x)


if __name__ == "__main__":
    unittest.main()
