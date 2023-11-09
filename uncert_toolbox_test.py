import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import uncertainty_toolbox as uct

import matplotlib.pyplot as plt


def generate_data():
    # generate ground truth data
    x = np.linspace(start=0, stop=10, num=400).reshape(-1, 1)
    y = np.squeeze(x * np.sin(x))

    # split in training, validation, test data
    x_train, x_valtest, y_train, y_valtest = train_test_split(
        x, y, test_size=0.5, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(
        x_valtest, y_valtest, test_size=0.5, random_state=42)

    # generate noisy training data
    noise_stddev = 0.4
    rng = np.random.RandomState(1)
    y_train += rng.normal(loc=0.0, scale=noise_stddev, size=y_train.shape)

    # generate validation and test data with different noise levels
    y_val_uniform_noisy = y_val + \
        rng.normal(loc=0.0, scale=4*noise_stddev, size=y_test.shape)
    y_test_uniform_noisy = y_test + \
        rng.normal(loc=0.0, scale=4*noise_stddev, size=y_test.shape)
    y_val_subrange_noisy = y_val_uniform_noisy.copy()
    y_test_subrange_noisy = y_test_uniform_noisy.copy()
    ind = [i for i in range(len(x_val)) if x_val[i] >= 4 and x_val[i] <= 7]
    y_val_subrange_noisy[ind] = y_val[ind] + \
        rng.normal(loc=0.0, scale=noise_stddev, size=len(ind))
    ind = [i for i in range(len(x_test)) if x_test[i] >= 4 and x_test[i] <= 7]
    y_test_subrange_noisy[ind] = y_test[ind] + \
        rng.normal(loc=0.0, scale=noise_stddev, size=len(ind))

    return {'x': x,
            'y': y,
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val_uniform_noisy': y_val_uniform_noisy,
            'y_val_subrange_noisy': y_val_subrange_noisy,
            'x_test': x_test,
            'y_test_uniform_noisy': y_test_uniform_noisy,
            'y_test_subrange_noisy': y_test_subrange_noisy
            }


def plot_gpr(ax, x, y, predict_mean, predict_stddev,
             x_train, y_train, x_test, y_test, title_str):

    ax.plot(x, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
    ax.scatter(x_train, y_train, label="training data")
    ax.scatter(x_test, y_test, edgecolors='blue',
               facecolors='None', label="test data")

    ax.plot(x, predict_mean, label="Mean prediction")
    ax.fill_between(x.ravel(),
                    predict_mean - 1.96 * predict_stddev,
                    predict_mean + 1.96 * predict_stddev,
                    color="tab:orange", alpha=0.5,
                    label=r"95% confidence interval",
                    )
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title(title_str)


def analyse_uncertainty(pred_mean_val, pred_std_val, y_val,
                        x_test, y_test, gaussian_process, axs):

    # run prediction on the test data
    recal_pred_mean, recal_pred_std = gaussian_process.predict(
        x_test, return_std=True)
    recal_pred_mean = recal_pred_mean.flatten()
    recal_pred_std = recal_pred_std.flatten()
    x_test = x_test.flatten()
    y_test = y_test.flatten()
    # get the expected proportions and observed proportions
    expected_props, observed_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        recal_pred_mean, recal_pred_std, y_test)

    # use isotonic regression for recalibration model.
    recal_model = uct.recalibration.iso_recal(expected_props, observed_props)

    # Get the expected proportions and observed proportions using the recalibrated model
    recal_exp_props, recal_obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        pred_mean_val, pred_std_val, y_val, recal_model=recal_model)

    y_val = y_val.flatten()
    uct.viz.plot_calibration(pred_mean_val, pred_std_val, y_val, ax=axs[0])
    axs[0].set_title("average calibration")
    uct.viz.plot_calibration(pred_mean_val, pred_std_val, y_val,
                             exp_props=recal_exp_props,
                             obs_props=recal_obs_props, ax=axs[1])
    axs[1].set_title("recalibrated average calibration")
    uct.viz.plot_adversarial_group_calibration(
        pred_mean_val, pred_std_val, y_val, ax=axs[2])
    axs[2].set_title("adversarial group calibration")


def run_example():
    # generate data
    data = generate_data()

    # fit gaussian process to the training data
    kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=12)
    gaussian_process.fit(data['x_train'], data['y_train'])
    pred_mean, pred_std = gaussian_process.predict(data['x'], return_std=True)

    # plot the predictions for the ground truth data together with the training and test data
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Gaussian process regression on a noisy dataset")
    plot_gpr(axs[0], data['x'], data['y'], pred_mean, pred_std,
             data['x_train'], data['y_train'],
             data['x_val'], data['y_val_uniform_noisy'], 'Uniform noise')
    plot_gpr(axs[1], data['x'], data['y'], pred_mean, pred_std,
             data['x_train'], data['y_train'],
             data['x_val'], data['y_val_subrange_noisy'], 'Subrange noise')

    pred_mean_val, pred_std_val = gaussian_process.predict(
        data['x_val'], return_std=True)

    # evaluate the calibration metrics and plot the results
    fig, axs = plt.subplots(2, 3)
    analyse_uncertainty(pred_mean_val, pred_std_val, data['y_val_uniform_noisy'],
                        data['x_test'], data['y_test_uniform_noisy'], gaussian_process, axs[0])
    analyse_uncertainty(pred_mean_val, pred_std_val, data['y_val_subrange_noisy'],
                        data['x_test'], data['y_test_subrange_noisy'], gaussian_process, axs[1])
    pad = 5
    ax = axs[0, 0]
    ax.annotate('uniform noise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')
    ax = axs[1, 0]
    ax.annotate('subrange noise', xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

    plt.show()


if __name__ == '__main__':
    run_example()
