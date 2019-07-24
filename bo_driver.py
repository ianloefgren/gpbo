#!/usr/bin/env python

"""
Driver script for Bayesian optimization of event-triggered decentralized
data fusion (ET-DDF).

Last modified: 7.24.2019
"""

import os
import sys
import yaml
import numpy as np

from ettdf.sim import SimInstance
from bayes_opt import BayesOpt, BayesOptModel, BayesOptSurrogate, BayesOptAcquisition

class ETDDFModel(BayesOptModel):

    def __init__(self):
        
        super().__init__()

        self.sim = SimInstance()

    def evaluate(self, design_point):

        delta = design_point[0]
        tau = design_point[1]

        self.sim.delta = delta
        self.sim.tau_state_goal = tau

        sim_result = self.sim.run_sim()
        cost = self.compute_cost(sim_result)

        return cost

    def compute_cost(self, sim_result):
        """
        Compute cost function for simulation using sim result structure.

        Parameters
        ------
        sim_result
            ET-DDF simulation result structure

        Returns
        -------
        cost
            computed cost of simulation run
        """

        pass


def main():
    
    # create Bayes Opt model instance containing ET-DDF sim
    etddf_model = ETDDFModel()

    # create Bayes Opt instance, specifying model and number of parameters
    opt_instance = BayesOpt(etddf_model, num_params=2)

    # compute optimal parameter values
    optimal_params = opt_instance.optimize()

    print(optimal_params)

if __name__ == "__main__":
    main()