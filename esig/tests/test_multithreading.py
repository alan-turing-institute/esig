import unittest
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import esig

from esig.tests.test_package_interface import ArrayTestCase


class TestMultiThreadingCalculations(ArrayTestCase):

    def setUp(self):
        rng = np.random.Generator(np.random.MT19937(12345))
        self.length = 100
        self.width = 5
        self.depth = 3
        self.data = rng.uniform(-4, 4, size=(self.length, self.width))

    @staticmethod
    def make_function(r, func):
        def wrap(data, depth):
            r = func(data, depth)
        return wrap

    def test_signature_parallel(self):

        with ThreadPoolExecutor(2) as pool:
            f1 = pool.submit(esig.stream2sig, self.data, self.depth)
            f2 = pool.submit(esig.stream2sig, self.data, self.depth)
            done, not_done = concurrent.futures.wait((f1, f2))

        self.assertFalse(not_done)
        self.assertEqual(len(done), 2)

        self.assert_allclose(f1.result(), f2.result())

    def test_log_signature_parallel(self):

        with ThreadPoolExecutor(2) as pool:
            f1 = pool.submit(esig.stream2logsig, self.data, self.depth)
            f2 = pool.submit(esig.stream2logsig, self.data, self.depth)
            done, not_done = concurrent.futures.wait((f1, f2))

        self.assertFalse(not_done)
        self.assertEqual(len(done), 2)

        self.assert_allclose(f1.result(), f2.result())


if __name__ == '__main__':
    unittest.main()
