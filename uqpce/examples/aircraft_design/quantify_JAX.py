import openmdao.api as om
import os
import numpy as np

from uqpce.mdao.uqpcegroup import UQPCEGroup
from uqpce.mdao import interface

from organize import initialize
from helpers import plot_objective, plot_coefficients, get_values

from disciplines_JAX.BreguetRangeCompJAX import BreguetRangeComp
from disciplines_JAX.aero_jax import AeroCompJax
from disciplines_JAX.total_mass_comp import TotalMassComp
from disciplines_JAX.propulsion import PropulsionComp
from disciplines_JAX.WeightsCompJAX import WeightsComp
from disciplines_JAX.engine_weight import EngineWeightComp
from disciplines_JAX.doc import DOC
from disciplines_JAX.dpm import Dpm

from fixed import parameters

class CoupledDisciplines(om.Group):

    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        # Aerodynamics Component
        self.add_subsystem(
            'Aero', AeroCompJax(vec_size=n),
            promotes_inputs=['S', 'AR', 'V_cruise',
                             'C_D0_base', 'ks_base', 'e_base', 'S_0',
                             'delta_CD0', 'delta_ks', 'delta_e',
                             'm_total'], 
            promotes_outputs=['CL', 'CD', 'LD', 'WL']
        )

        # Structural Weight Component
        self.add_subsystem(
            'Weight', WeightsComp(vec_size=n),
            promotes_inputs=['S', 'AR', 'V_cruise',
                             'kw_base', 'fsys_base', 'p_base',
                             'delta_kw', 'delta_fsys', 'delta_p',
                             'm_total', 'm_engine', 'm_fuse',
                             'V_ref'],
            promotes_outputs=['m_wing', 'm_empty']
        )

        # Total Mass Comp
        self.add_subsystem(
            'Mass', TotalMassComp(vec_size=n),
            promotes_inputs=['m_empty', 'm_fuel', 'm_payload'],
            promotes_outputs=['m_total']
        )

        # Breguet Range Component
        self.add_subsystem(
            'Range', BreguetRangeComp(vec_size=n),
            promotes_inputs=['V_cruise',
                             'm_total', 'LD',
                             'SFC',
                             'm_fuel'],
            promotes_outputs=['R']
        )

        # Range Residual
        initial_guess = np.ones(n)*20000 #kg
        Balance = om.BalanceComp()
        
        Balance.add_balance(
            name='m_fuel',
            val=initial_guess,
            units='kg',
            lower=1000.0,
            upper=100000.0,
            lhs_name='R',
            rhs_name='R_target',
            rhs_val=parameters['R_target'],
            eq_units='m',
            normalize=True,
            ref0=1000.0,
            ref=20000.0,
            res_ref=1.0,
        )
        
        self.add_subsystem(
            'Balance', Balance,
            promotes_inputs=['R', 'R_target'],
            promotes_outputs=['m_fuel']
        )
        
        # Residual Solver Options
        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['maxiter'] = 700
        self.nonlinear_solver.options['atol'] = 1e-8
        self.nonlinear_solver.options['rtol'] = 1e-8
        # newton.options['err_on_non_converge'] = True

        # See organize.py: ArmijoGoldstein backtracking was rejecting the full Newton
        # step nearly every iteration and collapsing convergence to linear. Bounds
        # enforcement alone restores quadratic convergence and is still needed to keep
        # the Jacobian non-singular.
        line_search = newton.linesearch = om.BoundsEnforceLS(bound_enforcement='vector')
        line_search.options['print_bound_enforce'] = True
        self.linear_solver = om.DirectSolver()

class CL_constraint(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        self.add_input('CL', units="unitless", shape=(n,))

        self.add_input('CL_target', val=0.53)

        self.add_output('CL_constraint', shape=(n,))

        # should be identity matrix
        self.declare_partials('CL_constraint', 'CL', rows=arange, cols=arange)
        self.declare_partials('CL_constraint', 'CL_target')

    def compute(self, inputs, outputs):

        CL = inputs['CL']
        CL_target = inputs['CL_target']

        outputs['CL_constraint'] = CL_target - CL

    def compute_partials(self, inputs, partials):

        partials['CL_constraint', 'CL'] = -1
        partials['CL_constraint', 'CL_target'] = 1

class WingLoad_constraint(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']
        arange = np.arange(n)

        self.add_input('WL', shape=(n,))

        self.add_input('WL_target', val=5905.0)

        self.add_output('WL_constraint', shape=(n,))

        # should be identity matrix
        self.declare_partials('WL_constraint', 'WL', rows=arange, cols=arange)

    def compute(self, inputs, outputs):

        WL = inputs['WL']
        WL_target = inputs['WL_target']

        outputs['WL_constraint'] = WL_target - WL

    def compute_partials(self, inputs, partials):

        partials['WL_constraint', 'WL'] = 1

def configure_subsystems(prob, vector_size=1):
    # Propulsion Component
    prob.model.add_subsystem(
        'Prop', 
        PropulsionComp(vec_size=vector_size),
        promotes_inputs=['V_cruise', 'SFC_tech',
                         'SFC_ref', 'V_ref',
                         'eta_base', 'kv_base',
                         'delta_eta', 'delta_kv'],
        promotes_outputs=['SFC']
    )

    # Engine Weight Component
    prob.model.add_subsystem(
        'Engine', 
        EngineWeightComp(vec_size=vector_size), 
        promotes_inputs=['SFC_tech', 
                         'm_eng_ref', 'alpha_base',
                         'delta_alpha'],
        promotes_outputs=['m_engine']
    )

    prob.model.add_subsystem(
        'AeroStruct', 
        CoupledDisciplines(vec_size=vector_size), 
        promotes_inputs=['V_cruise', 'S', 'AR',
                         'C_D0_base', 'ks_base', 'e_base', 'S_0',
                         'kw_base', 'fsys_base', 'p_base',
                         'V_ref', 'R_target', 'SFC',
                         'm_fuse', 'm_payload', 'm_engine',
                         'delta_CD0', 'delta_ks', 'delta_e',
                         'delta_fsys', 'delta_kw', 'delta_p'], 
        promotes_outputs=['m_fuel', 'm_empty', 'm_wing',
                          'm_total', 'LD', 'CL', 'CD', 'WL', 'R']
    )

   # prob.model.add_subsystem(
   #     'WingLoad_constraint', 
   #     WingLoad_constraint(vec_size=vector_size), 
   #     promotes_inputs=['WL'], 
   #     promotes_outputs=['WL_constraint']
   # )

    prob.model.add_subsystem(
        'LiftCoeff_constraint', 
        CL_constraint(vec_size=vector_size), 
        promotes_inputs=['CL'], 
        promotes_outputs=['CL_constraint']
    )

    prob.model.add_subsystem(
        'DOC_objective', 
        DOC(vec_size=vector_size), 
        promotes_inputs=['V_cruise', 'SFC_tech',
                         'Cf_base', 'beta_base',
                         'C_time', 'k_acq', 'C_eng_ref', 
                         'delta_beta', 'delta_Cf', 
                         'R', 'm_fuel'], 
        promotes_outputs=['DOC']
    )

    prob.model.add_subsystem(
        'DPM_objective', 
        Dpm(vec_size=vector_size), 
        promotes_inputs=['DOC', 'R', 'N_pax'], 
        promotes_outputs=['Dpm']
    )

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