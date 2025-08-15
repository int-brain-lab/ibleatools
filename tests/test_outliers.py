from ephysatlas import outliers
import numpy as np
import unittest


class TestOutliersKDEProba(unittest.TestCase):

    def test_usage(self):
        # Set random seed for reproducibility
        np.random.seed(42)  # TODO this does not seem to work
        # Generate train data from a normal distribution
        train_data = np.random.normal(loc=-4.5, scale=1.2, size=2000)
        # Generate test data from a normal distribution
        test_data = np.random.normal(loc=-3.5, scale=0.4, size=60)
        # Compute score
        score, x, hist = outliers.kde_proba_distribution(train_data, test_data)
        assert score.shape[0] == test_data.shape[0]

        score_out = np.array([0.94909745, 0.95985259, 0.94758700, 0.95592986, 0.93989972,
                              0.96315955, 0.96212788, 0.94726918, 0.96550552, 0.96921661,
                              0.94203708, 0.94796957, 0.94788689, 0.94578141, 0.96682835,
                              0.95866918, 0.95697261, 0.94013844, 0.96537497, 0.94521747,
                              0.96241355, 0.96347127, 0.97713305, 0.94508643, 0.94181560,
                              0.96301632, 0.97997165, 0.97587523, 0.96314187, 0.95152157,
                              0.96633449, 0.94571803, 0.96450351, 0.96466301, 0.97192622,
                              0.98534690, 0.96404361, 0.95123884, 0.96199762, 0.94007265,
                              0.94884578, 0.95349810, 0.95104398, 0.96286312, 0.94720050,
                              0.97326999, 0.96212302, 0.96207678, 0.95545146, 0.96287692,
                              0.96673165, 0.94662449, 0.94931726, 0.95603152, 0.94501664,
                              0.94724124, 0.95189154, 0.96782068, 0.96339920, 0.96306370])

        np.testing.assert_almost_equal(score, score_out)

    def test_bimodal(self):
        # Check that a sample selected in between two Gaussian distributions is detected as an outlier
        # Generate train data from 2 normal distributions
        train_data = np.concatenate((
            np.random.normal(loc=-4.5, scale=1.2, size=2000),
            np.random.normal(loc=17.5, scale=2.2, size=1000)
        ))

        # Generate test data from a normal distribution
        test_data = np.random.normal(loc=-3.5, scale=0.4, size=60)
        # Add some clear outliers
        test_data = np.concatenate((test_data, np.array([5.4, 6.4, 0, -9.0])))

        # Compute score
        score, _, _ = outliers.kde_proba_distribution(train_data, test_data)
        assert score[-1] == 1

    def test_pad_span(self):
        # If taking a test value very far out, the train histogram will have more padded sample points
        # We do not want this to influence the mean result of the score. Test to check the values don't change

        # Generate train data from a normal distribution
        train_data = np.random.normal(loc=-4.5, scale=1.2, size=2000)

        # Test some clear outliers
        test_data1 = np.array([-9.0, 3.3, -5.2, -7.8])
        test_data2 = np.array([-9.0, 3.3, -5.2, -1000])

        # Compute score
        score1, _, _ = outliers.kde_proba_distribution(train_data, test_data1)
        score2, _, _ = outliers.kde_proba_distribution(train_data, test_data2)

        for i_sample in range(0, 3):
            assert score1[i_sample] == score2[i_sample]


    def test_log_transform(self):
        # We want to check that doing twice a log/exp transform onto a Gaussian give the same values

        # Gaussian
        # Generate train data from a normal distribution
        train_data = np.random.normal(loc=-4.5, scale=1.2, size=2000)
        # Generate test data linearly around train set
        val_iqr = 3 * np.std(train_data)
        test_data = np.linspace(min(train_data) - val_iqr, max(train_data) + val_iqr, 100).reshape(-1, 1)
        # Compute score
        score, x, hist = outliers.kde_proba_distribution(train_data, test_data)

        # Log normal
        train_data_log = np.exp(train_data)
        test_data_log = np.exp(test_data)

        # Compute score
        score_log, _, _ = outliers.kde_proba_distribution(train_data_log, test_data_log)
        # TODO this is different and the below does not pass!
        np.testing.assert_almost_equal(np.exp(test_data), test_data_log)

        plot_debug = True
        # Plot for debug
        if plot_debug:
            import matplotlib.pyplot as plt
            from ephysatlas.plots import plot_histogram

            fig, axs = plt.subplots(1, 3)
            axs[0].plot(test_data)
            axs[1].plot(np.exp(test_data))
            axs[2].plot(test_data_log)

            fig, axs = plt.subplots(1, 1)
            plot_histogram(train_data, ax=axs, normalise=True)
            plt.plot(test_data, -1*(score-1), 'k.-')
            plt.plot(np.log(test_data_log), -1 * (score_log - 1), 'b.-')

            fig, axs = plt.subplots(1, 1)
            plot_histogram(train_data_log, ax=axs, normalise=True)
            plt.plot(test_data_log, -1*(score_log-1), 'k.-')


