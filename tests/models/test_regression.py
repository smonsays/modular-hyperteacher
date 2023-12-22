"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import unittest

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from metax.models import BayesianLinearRegression, RidgeRegression


class RegressionTestCase(unittest.TestCase):
    @staticmethod
    def generate_data(rng, num_features, num_samples, slope, intercept, stdv):
        rng_x, rng_eps = jax.random.split(rng)
        x = jax.random.uniform(rng_x, (num_samples, num_features))
        eps = stdv * jax.random.normal(rng_eps, (num_samples, ))

        weights = slope * jnp.ones((num_features))
        y = jnp.dot(weights, x.T) + intercept + eps

        return x, y

    def test_bayesian_regression(self):
        # Initialise random number generator
        rng = jax.random.PRNGKey(2022)
        rng_train, rng_test, rng_init, rng_call, rng_eval = jax.random.split(rng, 5)

        # Generate synthetic data
        x_train, y_train = self.generate_data(
            rng_train,
            num_features := 7,
            num_samples := 10000,
            slope := 3.0,
            intercept := 2.0,
            stdv := 0.3
        )
        x_test, y_test = self.generate_data(rng_test, num_features, 100, slope, intercept, stdv)

        # Initialise the model
        model = BayesianLinearRegression(
            num_features,
            a0 := 6,
            b0 := 6,
            lambda_prior := 0.25,
            use_intercept := bool(intercept)
        )
        params_init = model.init(rng_init, x_train)  # First rng remains unused internally

        # Fit on empty data
        x_empty = jnp.zeros((num_samples, num_features))
        y_empty = jnp.zeros((num_samples, ))
        params_empty = model.fit(params_init, x_empty, y_empty)
        assert all(jtu.tree_leaves(jtu.tree_map(jnp.allclose, params_empty, params_init)))

        @jax.jit
        def mse_loss(params, rng, input, target):
            pred = model.apply(params, rng, input)
            return jnp.mean(jnp.square(pred - target))

        # Fit and compute mean squared error
        loss_before = mse_loss(params_init, rng_call, x_test, y_test)
        params = model.fit(params_init, x_train, y_train)
        loss_after = mse_loss(params, rng_call, x_test, y_test)

        # Check if loss decreased and if slope and variance have been correctly approximated
        assert loss_before > loss_after
        assert all(jnp.isclose(params.mu[:-use_intercept], slope, atol=0.1))
        if use_intercept:
            assert jnp.isclose(params.mu[-1], intercept, atol=0.1)
        assert jnp.isclose(params.b / (params.a + 1), stdv**2, atol=0.1)

        # Fit on partially masked data
        x_train_masked = jnp.concatenate((x_train, jnp.zeros((masked_points := 100, num_features))))
        y_train_masked = jnp.concatenate((y_train, jnp.zeros((masked_points, ))))
        params_masked = model.fit(params_init, x_train_masked, y_train_masked)

        # Check that masking didn't change the resulting parameters
        assert all(jnp.isclose(params_masked.mu[:-use_intercept], slope, atol=0.1))
        if use_intercept:
            assert jnp.isclose(params.mu[-1], intercept, atol=0.1)
        assert jnp.isclose(params_masked.b / (params.a + 1), stdv**2, atol=0.1)

    def test_ridge_regression(self):
        # Initialise random number generator
        rng = jax.random.PRNGKey(2022)
        rng_train, rng_test, rng_init, rng_call, rng_eval = jax.random.split(rng, 5)

        # Generate synthetic data
        x_train, y_train = self.generate_data(
            rng_train,
            num_features := 7,
            num_samples := 10000,
            slope := 3.0,
            intercept := 5.0,
            stdv := 0.3
        )
        x_test, y_test = self.generate_data(rng_test, num_features, 100, slope, intercept, stdv)

        # Initialise the model
        model = RidgeRegression(
            num_features,
            l2_reg := 1.0,
            use_intercept := bool(intercept)
        )
        params_init = model.init(rng_init, x_train)  # First rng remains unused internally

        # Fit on empty data
        x_empty = jnp.zeros((num_samples, num_features))
        y_empty = jnp.zeros((num_samples, ))
        params_empty = model.fit(params_init, x_empty, y_empty)
        assert all(jtu.tree_leaves(jtu.tree_map(jnp.allclose, params_empty, params_init)))

        # Fit and compute mean squared error
        @jax.jit
        def mse_loss(params, input, target):
            pred = model.apply(params, input)
            return jnp.mean(jnp.square(pred - target))

        loss_before = mse_loss(params_init, x_test, y_test)
        params = model.fit(params_init, x_train, y_train)
        loss_after = mse_loss(params, x_test, y_test)

        # Check if loss decreased and if slope and variance have been correctly approximated
        assert loss_before > loss_after
        assert all(jnp.isclose(params.weight[:-use_intercept], slope, atol=0.1))
        if use_intercept:
            assert jnp.isclose(params.weight[-1], intercept, atol=0.1)

        # Fit on partially masked data
        x_train_masked = jnp.concatenate((x_train, jnp.zeros((masked_points := 100, num_features))))
        y_train_masked = jnp.concatenate((y_train, jnp.zeros((masked_points, ))))
        params_masked = model.fit(params_init, x_train_masked, y_train_masked)

        # Check that masking didn't change the resulting parameters
        assert all(jnp.isclose(params.weight[:-use_intercept], slope, atol=0.1))
        if use_intercept:
            assert jnp.isclose(params.weight[-1], intercept, atol=0.1)
