import openmdao.api as om
import numpy as np

from disciplines.BreguetRange import BreguetRangeComp
from disciplines.aero import AeroComp
from disciplines.total_mass_comp import TotalMassComp
from disciplines.propulsion import PropulsionComp
from disciplines.weight import EngineWeightComp, WeightsComp
from disciplines.doc import DOC
from disciplines.dpm import Dpm

from fixed import parameters, tuning

class CoupledDisciplines(om.Group):

    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        # Aerodynamics Component
        self.add_subsystem(
            'Aero', AeroComp(vec_size=n),
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

        # ArmijoGoldstein backtracking was rejecting the full Newton step on nearly
        # every iteration (residual halving at exactly 0.5), turning Newton into a
        # linear contraction and driving solves to ~700 iterations. Plain bounds
        # enforcement restores quadratic convergence; the bound handling itself is
        # still required (without any linesearch the Jacobian goes singular/NaN).
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

def initialize(prob, params=parameters):
    prob.set_val('V_cruise', params['V_cruise'])
    prob.set_val('S', params['S'])
    prob.set_val('AR', params['AR'])
    prob.set_val('SFC_tech', params['SFC_tech'])

    # Tuning Parameters
    prob.set_val('e_base', parameters['e_oswald_base'])
    prob.set_val('C_D0_base', parameters['CD0_base'])
    prob.set_val('Cf_base', parameters['Cf_base'])
    prob.set_val('fsys_base', tuning['fsys_base'])          # fraction of total mass comprising 'systems' and stuff
    prob.set_val('kw_base', tuning['kw_base'])              # wing weight regression/fit tuning parameter
    prob.set_val('p_base', tuning['p_base'])                # off (faster) design velocity wing weight penalty exponent parameter
    prob.set_val('eta_base', tuning['eta_base'])            # tuning paramter to change effect SFC_tech has on changing SFC_ref
    prob.set_val('kv_base', tuning['kv_base'])              # off design veloicty penalty to increase SFC qudratically about V_ref
    prob.set_val('beta_base', tuning['beta_base'])          # strength of increase/decrease of amortized engine cost due to SFC_tech
    prob.set_val('alpha_base', tuning['alpha_base'])        # strength of increase/decrease of engine mass due to SFC_tech
    prob.set_val('ks_base', tuning['ks_base'])              # pretty hard to estimate this. it represents the sensitivty 
                                                            # of the drag coefficient to changes in planform area linearized 
                                                            # about S_ref. I have no idea what to put for this, but I chose a 
                                                            # small value above. Note units are 1/m**2