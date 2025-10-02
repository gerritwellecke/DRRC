import unittest

import numpy as np
from sklearn.decomposition import PCA

from drrc.DR_Trafo_extension import (
    inversetransform_PCA,
    prepare_DRPCA_useLargest,
    transform_PCA,
)


class TestDR_Trafo_extension(unittest.TestCase):
    def test_prepare_DRPCA_useLargest(self):
        """
        Tests :code:`prepare_DRPCA_useLargest` function for 1d and 2d data.

        """
        ## 1d Test

        # Define your setup here
        data = np.array([[1, 0], [-1, 0], [2, 0], [-2, 0]], dtype=float)
        fraction_of_dimension = 0.5
        pca = PCA()
        pca.fit(data)

        # Call the function with the setup
        results = prepare_DRPCA_useLargest(data, fraction_of_dimension, None, None)

        # Define your expected output here
        expected_output0 = ...
        expected_output1 = [True, False]

        # Assert that the function output is as expected
        np.testing.assert_array_equal(results[0], expected_output0)
        np.testing.assert_array_equal(results[1], expected_output1)

        np.testing.assert_array_equal(pca.transform(data), results[2].transform(data))
        np.testing.assert_array_equal(pca.transform(data), results[3].transform(data))

        ## 2d Test
        # Define your setup here
        data = np.array([[[1, 0], [-1, 0]], [[2, 0], [-2, 0]]], dtype=float)
        pca = PCA()
        pca.fit(data.reshape(data.shape[0], -1))

        # Call the function with the setup
        results = prepare_DRPCA_useLargest(data, fraction_of_dimension, None, None)

        # Define your expected output here
        expected_output0 = ...
        expected_output1 = [True, True, False, False]

        # Assert that the function output is as expected
        np.testing.assert_array_equal(results[0], expected_output0)
        np.testing.assert_array_equal(results[1], expected_output1)

        np.testing.assert_array_equal(
            pca.transform(data.reshape(data.shape[0], -1)),
            results[2].transform(data.reshape(data.shape[0], -1)),
        )
        np.testing.assert_array_equal(
            pca.transform(data.reshape(data.shape[0], -1)),
            results[3].transform(data.reshape(data.shape[0], -1)),
        )

        pass

    def test_transform_PCA(self):
        """
        Tests :code:`transform_PCA` function for 1d and 2d data.
        """

        ## 1d Test
        # Define your setup here
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        pca = PCA(n_components=2)

        # Fit the PCA object to the data
        pca.fit(data)

        # Call the function with the setup
        result = transform_PCA(data, pca)

        # Define your expected output here
        expected_output = pca.transform(data.reshape(data.shape[0], -1))

        # Assert that the function output is as expected
        np.testing.assert_array_equal(result, expected_output)

        ## 2d Test
        # Define your setup here
        data = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 3, 5, 7], [2, 4, 6, 8]]], dtype=float
        )
        pca = PCA(n_components=2)

        # Fit the PCA object to the data
        pca.fit(data.reshape(data.shape[0], -1))

        # Call the function with the setup
        result = transform_PCA(data, pca)

        # Define your expected output here
        expected_output = pca.transform(data.reshape(data.shape[0], -1))

        # Assert that the function output is as expected
        np.testing.assert_array_equal(result, expected_output)

    def test_inversetransform_PCA(self):
        """
        Tests :code:`transform_PCA` function for 1d and 2d data.
        """

        ## 1d Test
        # Define your setup here
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        pca = PCA(n_components=2)

        # Fit the PCA object to the data
        pca.fit(data.reshape(data.shape[0], -1))

        # Call the function with the setup
        result = inversetransform_PCA(data[:, :2], pca)

        # Define your expected output here
        expected_output = pca.inverse_transform(data.reshape(data.shape[0], -1)[:, :2])

        # Assert that the function output is as expected
        np.testing.assert_array_equal(result, expected_output)

        ## 2d Test
        # Define your setup here
        data = np.array(
            [[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 3, 5, 7], [2, 4, 6, 8]]], dtype=float
        )
        pca = PCA()

        # Fit the PCA object to the data
        pca.fit(data.reshape(data.shape[0], -1))

        # Call the function with the setup
        result = inversetransform_PCA(data.reshape(data.shape[0], -1)[:, :2], pca)

        # Define your expected output here
        expected_output = pca.inverse_transform(data.reshape(data.shape[0], -1)[:, :2])

        # Assert that the function output is as expected
        np.testing.assert_array_equal(result, expected_output)
        pass


if __name__ == "__main__":
    unittest.main()
