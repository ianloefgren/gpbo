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

from etddf.sim import SimInstance
from etddf.helpers.config_handling import load_config
from bayes_opt import BayesOpt, BayesOptModel, BayesOptSurrogate, BayesOptAcquisition

class ETDDFModel(BayesOptModel):
    """
    Wrapper for ETDDF simulations to integrate into Bayes Opt framework. 
    Implements evaluate() method that Bayes Opt uses to sample the model.
    """

    def __init__(self,cfg):
        
        super().__init__()

        self.cfg = cfg

        self.fxn_evals = 0

    def evaluate(self, design_point):
        """
        Implementation of Bayes Opt model evaluate. Evalutes model at passed
        design point. Called in Bayes Opt optimize() fxn.
        """

        delta = design_point[0]
        tau = self.cfg['tau_values'][0]

        print('Starting simulation with params: delta={}, tau={}'.format(delta,tau))

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

        sim_result = sim.run_sim(['delta={}\ttau={}'.format(delta,tau),'Function evals: {}'.format(self.fxn_evals),''])
        cost = self.compute_cost(sim_result)

        self.fxn_evals += 1

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

        # extract data from results dictionary
        data = sim_result['results']

        agent_mse = []
        agent_comms_cost = []

        # find avg mse across agents and time
        for i in range(0,len(data['agents'])):
            
            # extract agent mse
            a = data['agents'][i]

            mse_history = a.mse_history

            # est_data = a.local_filter.state_history
            # cov_data = a.local_filter.cov_history
            # truth_data = a.true_state

            # extract baseline data to plot
            # b = data['baseline']
            # bl_est_data = b.state_history
            # bl_cov_data = b.cov_history

            # get agent location in agent and baseline data
            _, idx = a.get_location(a.agent_id)
            bl_idx = np.arange(a.agent_id*a.num_states,a.agent_id*a.num_states+a.num_states)

            # turn data lists of list into numpy array
            mse_data = np.array(mse_history)
            # mse_data = np.concatenate([np.array(x) for x in mse_history],axis=1)
            # est_data_vec = np.concatenate([np.array(x[idx]) for x in est_data],axis=1)
            # truth_data_vec = np.concatenate([np.expand_dims(x,axis=1) for x in truth_data],axis=1)
            # var_data_vec = np.concatenate([np.expand_dims(np.diag(x[np.ix_(idx,idx)]),axis=1) for x in cov_data],axis=1)

            # compute mean of agent mse
            mse_agent_mean = np.mean(mse_data)

            # compute comms cost
            comms_agent_cost = a.ci_trigger_cnt*a.local_filter.x.shape[0]**2 + a.ci_trigger_cnt*a.local_filter.x.shape[0] + a.msgs_sent

            agent_mse.append(mse_agent_mean)
            agent_comms_cost.append(comms_agent_cost)


        print('mse vals:')
        print(agent_mse)
        print('comms cost:')
        print(agent_comms_cost)

        # scale comms cost by factor of 1/100
        comms_cost = max(agent_comms_cost)/100
        mse_cost = max(agent_mse)

        cost = mse_cost + comms_cost

        return cost

class TestModel(BayesOptModel):

    def __init__(self):
        self.fxn = lambda x: (x-3)*(x+2)

    def evaluate(self,design_point):
        return self.fxn(design_point[0]) + np.random.normal(0,1)

class RastriginFxn1D(BayesOptModel):

    def __init__(self):
        self.fxn = lambda x: 10 + x**2 - 10*np.cos(2*np.pi*x)

    def evaluate(self,design_point):
        return self.fxn(design_point[0])


def main():
    
    # create Bayes Opt model instance containing ET-DDF sim
    cfg = load_config(os.path.abspath('/home/ian/Documents/school/grad/et-ddf/python/config/config.yaml'))
    etddf_model = ETDDFModel(cfg)

    # # create Bayes Opt instance, specifying model and number of parameters
    opt_instance = BayesOpt(etddf_model, param_bounds=[(0,10)])
    # opt_instance = BayesOpt(RastriginFxn1D(), param_bounds=[(-5.5,5.5)])
    # opt_instance = BayesOpt(TestModel(), param_bounds=[(-5.5,5.5)])

    # compute optimal parameter values
    optimal_params = opt_instance.optimize(max_iterations=20)

    print(optimal_params)

if __name__ == "__main__":
    main()