#!/usr/bin/env python

"""
Driver script for Bayesian optimization of event-triggered decentralized
data fusion (ET-DDF).

Last modified: 7.25.2019
"""

import os
import sys
import yaml
import numpy as np

# from ettdf.sim import SimInstance
# from etddf.helpers import load_config
from bayes_opt import BayesOpt, BayesOptModel, BayesOptSurrogate, BayesOptAcquisition

class ETDDFModel(BayesOptModel):
    """
    Wrapper for ETDDF simulations to integrate into Bayes Opt framework. 
    Implements evaluate() method that Bayes Opt uses to sample the model.
    """

    def __init__(self,cfg):
        
        super().__init__()

        self.cfg = cfg

    def evaluate(self, design_point):
        """
        Implementation of Bayes Opt model evaluate. Evalutes model at passed
        design point. Called in Bayes Opt optimize() fxn.
        """

        delta = design_point[0]
        tau = design_point[1]

        # create new simulation instance
        sim = SimInstance(delta=delta,tau=tau,msg_drop_prob=0,
                            baseline_cfg=self.cfg['baseline_cfg'],
                            agent_cfg=self.cfg['agent_cfg'],
                            max_time=self.cfg['max_time'],
                            dt=self.cfg['dt'],
                            use_adaptive_tau=self.cfg['use_adaptive_tau'],
                            fixed_rng=self.cfg['fixed_rng'],
                            process_noise=False,
                            sensor_noise=False)

        sim_result = sim.run_sim()
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

class TestModel(BayesOptModel):

    def __init__(self):
        self.fxn = lambda x: (x-3)*(x+2)

    def evaluate(self,design_point):
        return self.fxn(design_point[0])


def main():
    
    # create Bayes Opt model instance containing ET-DDF sim
    # cfg = load_config(os.path.abspath(os.path.join(
    #                     os.path.dirname(__file__),'config.yaml')))
    # etddf_model = ETDDFModel(cfg)

    # # create Bayes Opt instance, specifying model and number of parameters
    # opt_instance = BayesOpt(etddf_model, param_bounds=[(0,100),(0,100)])
    opt_instance = BayesOpt(TestModel(), param_bounds=[(-50,50)])

    # compute optimal parameter values
    optimal_params = opt_instance.optimize(max_iterations=100)

    print(optimal_params)

if __name__ == "__main__":
    main()