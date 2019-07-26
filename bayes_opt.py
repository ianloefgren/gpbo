#!/usr/bin/env python

"""
Implementation of Bayesian optimization using Gaussian processes (GPBO).

Last modified 7.24.2019
"""

import scipy
import numpy as np
import sklearn.gaussian_process
from scipy.optimize import minimize
from scipy.special import erf

class BayesOpt(object):

    def __init__(self, model, param_bounds, acquisition_fxn='EI', surrogate_fxn='GP-RBF', seed_data=None):
        
        # number of parameters to evaluate
        self.num_design_params = len(param_bounds)
        self.param_bounds = np.array(param_bounds)

        # save model, surrogate model, and acquisition fxn
        self.model = model
        self.surrogate = self.initialize_surrogate(surrogate_fxn,self.num_design_params, seed_data)
        self.acquisition = self.initialize_acquisition(acquisition_fxn)

        # best current design point that minimizes the acquisition fxn
        # stores best result as optimization runs
        self.best_current_minimizer = [(self.param_bounds[:,1]-self.param_bounds[:,0])*np.random.random(
                            self.param_bounds.shape[0]) + self.param_bounds[:,0],0]

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

    def initialize_surrogate(self, surrogate_fxn, num_design_params, seed_data=None):
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
            model = GPSurrogate(num_design_params)
            # model = sklearn.gaussian_process.G

            # seed surrogate w/ some uniformly sampled points
            if seed_data is None:
                X = (self.param_bounds[:,1]-self.param_bounds[:,0])*np.random.random(
                                (5,self.param_bounds.shape[0])) + self.param_bounds[:,0]
                y = []
                for i in range(0,X.shape[0]):
                    y.append(self.evaluate_model(X[i,:]))
                y = np.array(y)

            model.initialize(X,y)

        return model

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
        return model.update_parameters(sample_point,model_sample)

    def initialize_acquisition(self, acquisition_fxn):
        """
        Initialize acquisition function to be specified function.
        """
        if acquisition_fxn is 'EI':
            return ExpectedImprovement(self.param_bounds)

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

        return self.acquisition.evaluate(self.surrogate, best_current_minimizer)

    def optimize(self, tol=1E-5, max_iterations=1000):
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

        max_iterations : default 1000
            The number of iterations the optimization will run for until
            termination, regardless if an optimum is found.

        Returns
        -------
        optimal_design_point
            The optimal parameter choice found by the optimization.
        """

        num_iterations = 0
        diff = 999999999
        # compute first point to test w/ uniform sample of param space
        sample_point = (self.param_bounds[:,1]-self.param_bounds[:,0])*np.random.random(
                            self.param_bounds.shape[0]) + self.param_bounds[:,0]

        # while diff > tol and num_iterations < max_iterations:
        while num_iterations < max_iterations:

            # get next sample point from acquisition fxn
            sample_point = self.update_acquisition(sample_point,self.best_current_minimizer)
            # evaluate model at chosen sample point
            print('sample: {}'.format(sample_point))
            model_result = self.evaluate_model(sample_point)
            print('model result: {}'.format(model_result))
            
            if model_result < self.best_current_minimizer[1]:
                self.best_current_minimizer = [sample_point,model_result]
            
            # update the surrogate model with the new data point
            self.update_surrogate(self.surrogate, model_result, sample_point)

            num_iterations += 1
            # diff = scipy.linalg.norm(params - params_old)

        return self.best_current_minimizer

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

class ExpectedImprovement(BayesOptAcquisition):

    def __init__(self,param_bounds):

        super().__init__()

        self.pdf = lambda z: (1/np.sqrt(2*np.pi))*np.exp(-0.5*(z**2))
        self.cdf = lambda z: 1 - 0.5*(1+erf(z/np.sqrt(2)))
        self.std_norm = lambda q: mu
        self.param_bounds = param_bounds

    def acquisition_fxn(self,q,surrogate,best_current_minimizer):
        """
        Compute the acquisition fxn at design point q

        Parameters
        ----------
        q
            point at which to evaluate surrogate
        
        surrogate
            surrogate model

        best_current_minimizer : list
            best design point found so far, q*, [design point, value]

        Returns
        -------
        aq
            the value of the acquisition fxn at point q
        """

        mean, sigma = surrogate.evaluate([q])
        # print('mean: {}'.format(mean))
        # print('sigma: {}'.format(sigma))

        # compute standard normal of current design point q
        Z = (mean - best_current_minimizer[1]) / sigma

        if sigma > 0:
            aq = (mean - best_current_minimizer[1])*self.cdf(Z) + sigma*self.pdf(Z)
        else:
            aq = 0

        print('aq: {}'.format(aq))
        return aq

    def evaluate(self, surrogate, best_current_minimizer):

        next_sample_point = minimize(self.acquisition_fxn,
                                        best_current_minimizer[0],
                                        args=(surrogate,best_current_minimizer),
                                        method='L-BFGS-B',
                                        bounds=self.param_bounds)

        print('next sample point: {}'.format(next_sample_point.x))

        return next_sample_point.x

class GPSurrogate(BayesOptSurrogate):

    def __init__(self,num_params):

        super().__init__()

        self.gp = sklearn.gaussian_process.GaussianProcessRegressor()
        self.num_design_params = num_params
        self.data = np.empty((1,self.num_design_params))
        self.targets = np.empty((1,1))

    def evaluate(self,design_point):
        return self.gp.predict(design_point,return_std=True)
        
    def update_parameters(self,new_sample_point,sample_point_val):
        self.data[-1+1,:] = new_sample_point
        self.targets[-1+1] = sample_point_val
        self.gp.fit(self.data,self.targets)

    def initialize(self,data,targets):
        self.data = data
        self.targets = targets
        self.gp.fit(self.data,self.targets)


def main():
    pass

if __name__ == "__main__":
    main()