import openmdao.api as om
import os
import numpy as np

from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface
from openmdao.utils.assert_utils import assert_check_partials

from organize import configure_subsystems, initialize
from helpers import plot_objective, plot_coefficients, get_values, plot_pareto

def deterministic_optimization(prob):
    # Optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['maxiter'] = 1000
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['disp'] = True

    # Declare Design variables
    prob.model.add_design_var('S', lower=100.0, upper=180.0, ref=124.6)
    prob.model.add_design_var('AR', lower=7.0, upper=50.0, ref=9.45)
    prob.model.add_design_var('V_cruise', lower=200, upper=260, ref=1)
    prob.model.add_design_var('SFC_tech', lower=-1, upper=1, ref=1)

    # Declare Objective Function
    prob.model.add_objective('DOC', ref=1.0e4)
    
    # prob.model.add_constraint('m_fuel', lower=1000.0, upper=50000.0, ref=16000.0)
    prob.model.add_constraint('CL', upper=0.6055, ref=0.1)
    # determ_prob.model.add_constraint('WL_constraint', lower=-5905, upper=5905, ref=0.1)

    prob.setup()
    initialize(prob)

    prob.run_driver()
    print(prob.get_val('DOC'))
    # display_results(determ_prob)
    # partial_data = prob.check_partials(out_stream=None, method='cs')
    # assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    S_opt = prob.get_val('S')
    AR_opt = prob.get_val('AR')
    V_cruise_opt = prob.get_val('V_cruise')
    SFC_tech_opt = prob.get_val('SFC_tech')

    optimal = {
       'V_cruise': V_cruise_opt,
       'AR': AR_opt,
       'S': S_opt,
       'SFC_tech': SFC_tech_opt
    }

    return optimal

def generate_output_list():
    probabilistic_Dpm_list = ['Dpm:resampled_responses','Dpm:ci_lower',
                              'Dpm:ci_upper','Dpm:mean','Dpm:variance']
    
    probabilistic_m_fuel_list = ['m_fuel:resampled_responses','m_fuel:ci_lower',
                                 'm_fuel:ci_upper','m_fuel:mean','m_fuel:variance']
    
    probabilistic_m_empty_list = ['m_empty:resampled_responses','m_empty:ci_lower',
                                  'm_empty:ci_upper', 'm_empty:mean','m_empty:variance']
    
    probabilistic_m_engine_list = ['m_engine:resampled_responses','m_engine:ci_lower',
                                   'm_engine:ci_upper','m_engine:mean','m_engine:variance']
    
    probabilistic_m_total_list = ['m_total:resampled_responses','m_total:ci_lower',
                                  'm_total:ci_upper','m_total:mean','m_total:variance']
    
    probabilistic_CL_list = ['CL:resampled_responses','CL:ci_lower',
                             'CL:ci_upper','CL:mean','CL:variance']

    probabilistic_CD_list = ['CD:resampled_responses','CD:ci_lower',
                             'CD:ci_upper','CD:mean','CD:variance']
    
    probabilistic_SFC_list = ['SFC:resampled_responses','SFC:ci_lower',
                              'SFC:ci_upper','SFC:mean','SFC:variance']
    
    probabilistic_CL_constr_list = ['CL_constraint:resampled_responses',
                                    'CL_constraint:ci_lower',
                                    'CL_constraint:ci_upper',
                                    'CL_constraint:mean',
                                    'CL_constraint:variance']
    
    # probabilistic_WL_constr_list = ['WL_constraint:resampled_responses',
    #                                 'WL_constraint:ci_lower',
    #                                 'WL_constraint:ci_upper',
    #                                 'WL_constraint:mean',
    #                                 'WL_constraint:variance']

    probabilistic_output_list =  (
        probabilistic_Dpm_list +
        probabilistic_m_fuel_list +
        probabilistic_m_empty_list +
        probabilistic_m_engine_list +
        probabilistic_m_total_list +
        probabilistic_CL_list +
        probabilistic_CD_list +
        probabilistic_SFC_list +
        probabilistic_CL_constr_list
    )
    
    return probabilistic_output_list

class Uncertain_Objective(om.ExplicitComponent):
    
    def setup(self):
        # Proposed Design Variables
        self.add_input('DOC:mean', units='USD')
        self.add_input('DOC:variance', units='USD**2')
        self.add_input('lambda', val = 0, units="unitless")

        #scaling quantites
        self.add_input('DOC:mean_resp', val=1.0, units='USD')
        self.add_input('DOC:var_resp', val=1.0, units='USD**2')
        
        # Outputs
        self.add_output('DOC:mean_plus_lambda_variance', units='unitless')
       
    def setup_partials(self):
        self.declare_partials('DOC:mean_plus_lambda_variance', ['DOC:mean', 'DOC:variance', 'lambda', 'DOC:mean_resp', 'DOC:var_resp'])

    def compute(self, inputs, outputs):
        lambd = inputs['lambda']
        var = inputs['DOC:variance']
        mu = inputs['DOC:mean']

        var_resp = inputs['DOC:var_resp']
        mu_resp = inputs['DOC:mean_resp']

        outputs['DOC:mean_plus_lambda_variance'] = (mu/mu_resp) + lambd * (var/var_resp)

    def compute_partials(self, inputs, partials):
        var = inputs['DOC:variance']
        mu = inputs['DOC:mean']
        lambd = inputs['lambda']
        var_resp = inputs['DOC:var_resp']
        mu_resp = inputs['DOC:mean_resp']

        partials['DOC:mean_plus_lambda_variance','DOC:variance'] = lambd/var_resp
        partials['DOC:mean_plus_lambda_variance','DOC:mean'] = 1.0/mu_resp
        partials['DOC:mean_plus_lambda_variance','lambda'] = var/var_resp
        partials['DOC:mean_plus_lambda_variance','DOC:mean_resp'] = -mu/mu_resp**2
        partials['DOC:mean_plus_lambda_variance','DOC:var_resp'] = -lambd * (var/var_resp**2)

def main():
    # Fix the numpy RNG so the UQPCE resampling (and thus the optimization path)
    # is repeatable from run to run.
    np.random.seed(0)

    #---------------------------------------------------------------------------
    #                      Run Deterministic Optimization
    #---------------------------------------------------------------------------

    determ_prob = om.Problem()

    configure_subsystems(determ_prob)

    optimal = deterministic_optimization(determ_prob)
    
    #---------------------------------------------------------------------------
    #                               Input Files
    #---------------------------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_yaml = 'input.yaml'
    relative_matrix = 'run_matrix_generated.dat'
    input_file = os.path.join(script_dir, relative_yaml)
    matrix_file = os.path.join(script_dir, relative_matrix)

    #---------------------------------------------------------------------------
    #             Setting up for UQPCE and Design Under Uncertainty
    #---------------------------------------------------------------------------

    (var_basis, norm_sq, resampled_var_basis, 
     aleatory_cnt, epistemic_cnt, resp_cnt, 
     order, variables, sig, run_matrix ) = interface.initialize(input_file, 
                                                                matrix_file)
    
    uncertain_prob = om.Problem()
    configure_subsystems(uncertain_prob,vector_size=resp_cnt)

    uncertain_prob.driver = om.ScipyOptimizeDriver()
    uncertain_prob.driver.options['optimizer'] = 'SLSQP'
    uncertain_prob.driver.options['maxiter'] = 1000
    uncertain_prob.driver.options['tol'] = 1e-6
    uncertain_prob.driver.options['disp'] = True

    #---------------------------------------------------------------------------
    #                       Add UQPCE Group to Problem
    #---------------------------------------------------------------------------

    probabilistic_DOC_output_list = ['DOC:resampled_responses','DOC:ci_lower',
                                     'DOC:ci_upper','DOC:mean','DOC:variance']
    other_output_list = generate_output_list()
    probabilistic_output_list = probabilistic_DOC_output_list + other_output_list

    uncertain_prob.model.add_subsystem(
        'UQPCE',
        
        UQPCEGroup(significance=sig,
                   var_basis=var_basis,
                   norm_sq=norm_sq,
                   resampled_var_basis=resampled_var_basis,
                   tail='both',
                   epistemic_cnt=epistemic_cnt,
                   aleatory_cnt=aleatory_cnt,
                   uncert_list=['DOC','Dpm', 'm_fuel','m_empty',
                                'm_engine','m_total','CL',
                                'CD','SFC','CL_constraint'],
                   tanh_omega=1e-3,
                   sample_ref0=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   sample_ref=[5.0e4, 1e-2, 1e3, 1e3, 1e3, 1e3, 0.1, 0.1, 1e-4, 0.1]),
        
        promotes_inputs=['DOC','Dpm', 'm_fuel','m_empty','m_engine',
                         'm_total','CL','CD','SFC','CL_constraint'],

        promotes_outputs=probabilistic_output_list
    )

    #---------------------------------------------------------------------------
    #                           Add Design Variables
    #---------------------------------------------------------------------------

    uncertain_prob.model.add_design_var('S',lower=100.0,upper=180.0,
                                        ref0=100.0,ref=180.0)

    uncertain_prob.model.add_design_var('AR',lower=7.0,upper=50.0,
                                        ref0=7.0,ref=50.0)

    uncertain_prob.model.add_design_var('V_cruise',lower=200.0,upper=260.0,
                                        ref0=200.0,ref=260.0)

    uncertain_prob.model.add_design_var('SFC_tech',lower=-1.0,upper=1.0,
                                        ref0=-1.0,ref=1.0)

    #---------------------------------------------------------------------------
    #                             Add Constraints
    #---------------------------------------------------------------------------

    uncertain_prob.model.add_constraint('CL:ci_upper',upper=0.65, ref0=0, ref=0.56)
    # uncertain_prob.model.add_constraint('DOC:ci_upper', upper=69500)

    #---------------------------------------------------------------------------
    #                      Add Probability-Based Objective
    #                       To Optimize Under Uncertainty          
    #---------------------------------------------------------------------------

    uncertain_prob.model.add_subsystem(
        'variable_risk_objective', Uncertain_Objective(),
        promotes_inputs=['DOC:mean', 'DOC:variance', 'lambda', 'DOC:mean_resp', 'DOC:var_resp'],
        promotes_outputs=['DOC:mean_plus_lambda_variance']
    )

    uncertain_prob.model.add_objective('DOC:mean_plus_lambda_variance', ref=120e3)
    # uncertain_prob.model.add_objective('DOC:mean', ref=6e4)    
    
    #---------------------------------------------------------------------------
    #                       Compute Model Response at 
    #                         Deterministic Optima      
    #---------------------------------------------------------------------------

    uncertain_prob.setup()

    initialize(uncertain_prob, params=optimal)
    
    interface.set_vals(uncertain_prob, variables, run_matrix)

    uncertain_prob.run_model()
    # interface.analysis(uncertain_prob, 'DOC', 'input.yaml', 'run_matrix_generated.dat')

    #plot_uqpce_pretty(uncertain_prob)

    response = get_values(uncertain_prob, copybool=True)

    #print(response)

    print("Objective Response from Run Model:")
    print(uncertain_prob.get_val('DOC:mean_plus_lambda_variance'))
    print("Should eaqual", uncertain_prob.get_val('DOC:mean'))

    #---------------------------------------------------------------------------
    #                      Reset Constraints Based on Response              
    #---------------------------------------------------------------------------
    # calculated_bound = response["CL"]["ci_upper"]
    # uncertain_prob.model.set_constraint_options('CL:ci_upper',upper=calculated_bound)
    
    #---------------------------------------------------------------------------
    #                      Optimize DOC Under Uncertainty              
    #---------------------------------------------------------------------------
    initialize(uncertain_prob)

    mean_response = response["DOC"]["mu"]
    variance_response = response["DOC"]["variance"]

    lambd_50 = mean_response/variance_response
    uncertain_prob.model.set_val('lambda', lambd_50)

    #vary lambda from 0.2 lambda_50 to 1.8 lambda_50

    #uncertain_prob.set_val('DOC:mean_resp', mean_response)
    #uncertain_prob.set_val('DOC:var_resp', variance_response)

    uncertain_prob.run_driver()

    # partial_data = uncertain_prob.check_partials(out_stream=None, method='cs')
    # assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)

    print(uncertain_prob.get_val('S'))
    print(uncertain_prob.get_val('AR'))
    print(uncertain_prob.get_val('V_cruise'))
    print(uncertain_prob.get_val('SFC_tech'))
    print(lambd_50)

    optimized = get_values(uncertain_prob)

    print(optimized["DOC"]["mu"])
    print(optimized["DOC"]["variance"])

    # plot_pareto(uncertain_prob, lambd_50)
    # print(optimal)

    #---------------------------------------------------------------------------
    #                  Plot Results and Compare Distributions              
    #---------------------------------------------------------------------------

    # print(uncertain_prob.get_val('R'))

    plot_objective(response, optimized)

    plot_coefficients(response, optimized)
    
    # plot_constraints(response, optimized)

    # plot_mass(response, optimized)

    # plot_sfc(response, optimized)
    #print("Response\n")
    #print(response["Design"])
    #print("Uncertain\n")
    #print(optimized["Design"])

if __name__ == "__main__":
    main()