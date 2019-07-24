#!/usr/bin/env python

"""
Implementation of Bayesian optimization using Gaussian processes (GPBO).

Last modified 7.24.2019
"""

import scipy
import numpy as np
import sklearn.gaussian_process

class BayesOpt(object):

    def __init__(self, model, num_params, acquisition_fxn='EI', surrogate_fxn='GP-RBF', seed_data=None):
        
        # number of parameters to evaluate
        self.num_design_params = 2

        # save model, surrogate model, and acquisition fxn
        self.model = model
        self.surrogate = self.initialize_surrogate(surrogate_fxn,seed_data)
        self.aquisition = self.initialize_acquisition(acquisition_fxn)

        best_current_minimizer = None

    def evaluate_model(self, sample_point):
        """
        Evaluate the black box model at the selected sample point.

        Parameters
        ----------
        sample_point
            Selected point to evaluate the model.

        Returns
        -------
        model_sample
            Result of sampling the model at the selected sample point.
        """
        return self.model.evaluate(sample_point)

    def initialize_surrogate(self, surrogate_fxn, seed_data=None):
        """
        Initialize the surrogate model to be used.

        Parameters
        ----------
        surrogate_fxn
            string specifying type of surrogate model
            In the case of a GP, kernel type follows a hyphen, Ex. 'GP-RBF'

        Returns
        -------
        None
        """
        if 'GP' in surrogate_fxn:
            # extract kernel type
            kernel = surrogate_fxn.split('-')[1]

            # create surrogate model instance
            model = GPSurrogate(kernel=kernel, seed=seed_data)

    def update_surrogate(self, model, model_sample, sample_point):
        """
        Update the surrogate model and parameters with the result of the
        last black box model evaluation.

        Parameters
        ----------
        model
            Surrogate model object to be updated.
        model_sample
            The result of the last black box model evaluation.
        sample_point
            The design point at which the black box model was last sampled.

        Returns
        -------
        updated_model
            Updated surrogate model.
        """
        pass

    def initialize_acquisition(self, acquisition_fxn):
        """
        Initialize acquisition function to be specified function.
        """
        pass

    def update_acquisition(self, sample_point, best_current_minimizer):
        """
        Compute the next sample location for the black box model using the
        chosen acquisition function and optimization routine.

        Parameters
        ----------
        sample_point
            Design point where the black box model was last evaluated.

        best_current_minimizer
            The best current design point that minimizes the objective
            function.

        Returns
        -------
        new_sample_point
            The new best point at which to sample the black box model.
        """

        pass

    def optimize(self, tol=1E-5, max_iterations=10000):
        """
        Find the optimal design point using Bayesian optimization.

        Parameters
        ----------
        tol : default 1E-5
            The convergence tolerance for the design parameters. The
            optimization ends if the parameters change by less than this
            amount between iterations. 
            
            Set to -1 to not use tolerance as a
            stopping criteria.

        max_iterations : default 10000
            The number of iterations the optimization will run for until
            termination, regardless if an optimum is found.

        Returns
        -------
        optimal_design_point
            The optimal parameter choice found by the optimization.
        """

        num_iterations = 0
        diff = 999999999
        sample_point = 

        while diff > tol and num_iterations < max_iterations:

            sample_point = self.update_acquisition(sample_point)
            model_result = self.evaluate_model(sample_point)
            self.update_surrogate(self.model, model_result, sample_point)

            num_iterations += 1
            diff = scipy.linalg.norm(params - params_old)

        return optimal_design_point

class BayesOptSurrogate:

    def __init__(self):
        pass

    def evaluate():
        pass

    def update_parameters():
        pass

class BayesOptAcquisition:

    def __init__(self):
        pass

    def get_best_design_point(self):
        raise NotImplementedError

class BayesOptModel:

    def __init__(self):
        pass

    def evaluate(self):
        raise NotImplementedError


class ExpectedImprovementAcquisition(BayesOptAcquisition):

    def __init__(self):

        super().__init__()

        self.cdf = lambda z: djlkfs
        self.pdf = lambda z: fjslkd
        self.std_norm = lambda q: mu

    def get_best_design_point(self):

        # compute standard normal of current design point q
        Z = (mu - obj_fxn_best) / sigma

        if sigma > 0:
            aq = (mu - best_minimizer)*self.cdf(Z) + sigma*self.pdf(Z)
        else:
            aq = 0

        return aq

def main():
    pass

if __name__ == "__main__":
    main()