import numpy as np
import pymc.math as pmath
from pymc import Model, Deterministic
import pystan
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.linear_model import LinearRegression

tfd = tfp.distributions

class ProbabilisticReasoningModule:
    def __init__(self, model_type='normal', priors=None, regularization=None, PPL='PyMC3'):
        self.model_type = model_type
        self.priors = priors if priors else {'mu': ('Uniform', -10, 10), 'sigma': ('HalfNormal', 1)}
        self.regularization = regularization
        self.PPL = PPL
        self.probabilistic_programming_language = PPL  # Add this line

    def preprocess_data(self, data):
        clean_data = data[np.isfinite(data)]
        normalized_data = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        return normalized_data

    def build_model(self, X, y):
        if self.PPL == 'Stan':
            stan_model_code = """
            data {
                int<lower=0> N;
                vector[N] x;
                vector[N] y;
            }
            parameters {
                real alpha;
                real beta;
                real<lower=0> sigma;
            }
            model {
                y ~ normal(alpha + beta * x, sigma);
            }
            """
            stan_model = pystan.StanModel(model_code=stan_model_code)
            stan_data = {'N': len(X), 'x': X, 'y': y}
            fit = stan_model.sampling(data=stan_data, iter=2000, chains=4)
            return fit
        elif self.PPL == 'PyMC3':
            with pm.Model() as model:
                alpha = pm.Normal('alpha', mu=0, sigma=10)
                beta = pm.Normal('beta', mu=0, sigma=10)
                sigma = pm.HalfNormal('sigma', sigma=1)
                mu = alpha + beta*X
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
            return model

    def perform_inference(self, model, X, y, inference_type='mcmc', sampler='Metropolis', cores=1, batch_size=None):
        if self.PPL == 'Edward':
            model = tfd.JointDistributionSequential([
                tfd.Normal(loc=0., scale=10., name="alpha"),  
                tfd.Normal(loc=0., scale=10., name="beta"),  
                tfd.HalfNormal(scale=1., name="sigma"),  
                lambda sigma, beta, alpha: tfd.Normal(loc=alpha + beta * X, scale=sigma, name="y")  
            ])
            sample = model.sample()
            conditioned_model = model.experimental_pin(y=y)
            surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
                model.event_shape_tensor(),
                model.default_event_space_bijector(),
                initializer=tf.compat.v1.glorot_uniform_initializer())
            losses = tfp.vi.fit_surrogate_posterior(
                conditioned_model.unnormalized_log_prob,
                surrogate_posterior=surrogate_posterior,
                optimizer=tf.optimizers.Adam(learning_rate=0.1),
                num_steps=1000)
            posterior_samples = surrogate_posterior.sample(1000)
            return posterior_samples
        elif self.PPL == 'PyMC3':
            with model:
                if inference_type.lower() == 'mcmc':
                    if sampler == 'NUTS':
                        trace = pm.sample(2000, tune=1000, cores=cores)
                    elif sampler == 'Metropolis':
                        step = pm.Metropolis()
                        trace = pm.sample(2000, tune=1000, step=step, cores=cores)
                    elif sampler == 'Hamiltonian':
                        step = pm.HamiltonianMC()
                        trace = pm.sample(2000, tune=1000, step=step, cores=cores)
                elif inference_type.lower() == 'advi':
                    approx = pm.fit(n=30000, method='advi')
                    trace = approx.sample(draws=5000)
                return trace

    def evaluate_model(self, trace):
        pm.traceplot(trace)
        pm.summary(trace).round(2)

    def predict(self, model, trace, X_new):
        with model:
            pp_samples = pm.sample_posterior_predictive(trace, vars=[model.Y_obs], samples=2000)
        y_pred_mean = np.mean(pp_samples['Y_obs'], axis=0)
        y_pred_std = np.std(pp_samples['Y_obs'], axis=0)
        return y_pred_mean, y_pred_std

    def tune_hyperparameters(self, hyperparameters, X, y):
        best_waic = np.inf
        best_hyperparameters = None
        best_model = None
        best_trace = None
        for hyperparameter in hyperparameters:
            with pm.Model() as model:
                alpha = pm.Normal('alpha', mu=hyperparameter[0], sigma=10)
                beta = pm.Normal('beta', mu=hyperparameter[1], sigma=10)
                sigma = pm.HalfNormal('sigma', sigma=1)
                mu = alpha + beta*X
                Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
                trace = pm.sample(2000, tune=1000)
                waic = pm.waic(trace)
                if waic < best_waic:
                    best_waic = waic
                    best_hyperparameters = hyperparameter
                    best_model = model
                    best_trace = trace
        return best_model, best_trace, best_hyperparameters

    def compare_models(self, models):
        waics = {}
        for i, model in enumerate(models):
            with model:
                trace = pm.sample(2000, tune=1000)
                waic = pm.waic(trace)
                waics[f"model_{i+1}"] = waic
        min_waic = min(waics.values())
        waic_diffs = {model: waic - min_waic for model, waic in waics.items()}
        return waic_diffs

    def integrate_ml(self, model, X, y):
        lr = LinearRegression().fit(X, y)
        with pm.Model() as model:
            alpha = pm.Normal('alpha', mu=lr.intercept_, sigma=1)
            beta = pm.Normal('beta', mu=lr.coef_, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = alpha + beta*X
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(2000, tune=1000)
        return model, trace

    def online_learning(self, model, trace, X_new, y_new):
        with model:
            pp_samples = pm.sample_posterior_predictive(trace, vars=[model.Y_obs], samples=2000)
        with pm.Model() as new_model:
            alpha = pm.Normal('alpha', mu=0, sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=10)
            sigma = pm.HalfNormal('sigma', sigma=1)
            mu = alpha + beta*X_new
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=pp_samples['Y_obs'])
            new_trace = pm.sample(2000, tune=1000)

        return new_model, new_trace

    def evaluate_expression(self, data):
        data = self.preprocess_data(data)
        model = self.build_model(*data)  # assuming data is a tuple (X, y)
        trace = self.perform_inference(model, *data)  # assuming data is a tuple (X, y)
        self.evaluate_model(trace)
        return trace