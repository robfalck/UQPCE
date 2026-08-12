import openmdao.api as om 
import numpy as np
from fixed import parameters

class WeightsComp(om.ExplicitComponent):
    """
    Component for "WeightsComp" box containing analytical derivatives
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n= self.options['vec_size']

        #proposed design variables
        self.add_input('S', units='m**2')
        self.add_input('AR', units="unitless")
        self.add_input('V_cruise', units='m/s')

        #model variable (output from other component)
        self.add_input('m_total', units='kg', shape=(n,))
        self.add_input('m_engine', units='kg', shape=(n,))

        #uncertain parameters
        self.add_input("delta_kw", val=np.ones(n), units="unitless", shape=(n,)) 
        self.add_input("delta_fsys", val=np.ones(n), units="unitless", shape=(n,))
        self.add_input("delta_p", val=np.ones(n), units="unitless", shape=(n,))

        #tuning parameters
        self.add_input('kw_base', units="unitless")
        self.add_input('fsys_base', units="unitless")
        self.add_input('p_base', units="unitless")
        
        #constant parameters
        self.add_input('V_ref', val=parameters['V_ref'], units='m/s')
        self.add_input('m_fuse', val=parameters['m_fuse'], units='kg')

        #outputs
        # res_ref scales each residual to O(1) for the AeroStruct Newton solve;
        # m_empty (~1.5e4 kg) was 87% of the initial residual norm on its own.
        self.add_output('m_empty', units='kg', shape=(n,), res_ref=1.0e4)
        self.add_output('m_wing', units='kg', shape=(n,), res_ref=1.0e3)

    def setup_partials(self):
        n = self.options['vec_size']

        indices = np.arange(n)
        scalar_columns = np.zeros(n, dtype=int)

        m_wing_scalar_inputs = ['S','AR','V_cruise','kw_base','p_base','V_ref']
        m_wing_vector_inputs = ['m_total','delta_kw','delta_p']

        m_empty_scalar_inputs = ['S','AR','V_cruise','kw_base','fsys_base','p_base','V_ref','m_fuse']
        m_empty_vector_inputs = ['m_total','m_engine','delta_kw','delta_fsys','delta_p']

        self.declare_partials('m_wing',m_wing_scalar_inputs,rows=indices,cols=scalar_columns)

        self.declare_partials('m_wing',m_wing_vector_inputs,rows=indices,cols=indices)

        self.declare_partials('m_empty',m_empty_scalar_inputs,rows=indices,cols=scalar_columns)

        self.declare_partials('m_empty',m_empty_vector_inputs,rows=indices,cols=indices)

    def compute(self, inputs, outputs):
        """
        m_wing = kw_base · S^0.758 · AR^0.6 · m_total^0.006 · (V/V_ref)^p_base

        m_empty = m_wing + m_fuse + fsys_base · m_total + m_engine
        """

        S = inputs['S']
        AR = inputs['AR']
        V = inputs['V_cruise']

        m_total = inputs['m_total']
        m_engine = inputs['m_engine']

        delta_kw = inputs['delta_kw']
        delta_fsys = inputs['delta_fsys']
        delta_p = inputs['delta_p']
        
        kw_base = inputs['kw_base']
        fsys_base = inputs['fsys_base']
        p_base = inputs['p_base']
        V_ref = inputs['V_ref']
        m_fuse = inputs['m_fuse']

        m_wing = (kw_base * delta_kw * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p)))

        outputs['m_wing'] = m_wing

        outputs['m_empty'] = m_wing + m_fuse + (fsys_base * m_total* delta_fsys) + m_engine 

    def compute_partials(self, inputs, partials):
        
        kw_base = inputs['kw_base']
        fsys_base = inputs['fsys_base']
        p_base = inputs['p_base']
        V_ref = inputs['V_ref']

        S = inputs['S']
        AR = inputs['AR']
        V = inputs['V_cruise']

        m_total = inputs['m_total']

        delta_kw = inputs['delta_kw']
        delta_fsys = inputs['delta_fsys']
        delta_p = inputs['delta_p']

        m_wing = (kw_base * delta_kw * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p)))
        
        partials['m_wing', 'S'] = 0.758 * m_wing / S
        partials['m_wing', 'AR'] = 0.6 * m_wing / AR
        partials['m_wing', 'm_total'] = 0.006 * m_wing / m_total
        partials['m_wing', 'V_cruise'] = delta_p * p_base * m_wing / V

        partials['m_wing', 'delta_kw'] = kw_base * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p))
        partials['m_wing', 'delta_p'] = m_wing * p_base * np.log(V / V_ref)
        partials['m_wing', 'kw_base'] = delta_kw * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p))

        partials['m_wing', 'V_ref'] = -(p_base * delta_p) * m_wing / V_ref
        partials['m_wing', 'p_base'] = m_wing * delta_p * np.log(V/V_ref)

        #m_empty grads

        partials['m_empty', 'S'] = 0.758 * m_wing / S
        partials['m_empty', 'AR'] = 0.6 * m_wing / AR
        partials['m_empty', 'V_cruise'] = (p_base * delta_p) * m_wing / V

        partials['m_empty', 'm_total'] = 0.006 * m_wing / m_total + fsys_base * delta_fsys
        partials['m_empty', 'm_engine'] = 1.0

        partials['m_empty', 'delta_kw'] = kw_base * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p))
        partials['m_empty', 'delta_fsys'] = fsys_base * m_total
        partials['m_empty', 'delta_p'] = m_wing * p_base * np.log(V/V_ref)

        partials['m_empty', 'kw_base'] = delta_kw * (S ** 0.758) * (AR ** 0.6) * (m_total ** 0.006) * ((V/V_ref) ** (p_base * delta_p))
        partials['m_empty', 'fsys_base'] = m_total * delta_fsys

        partials['m_empty', 'p_base'] = m_wing * delta_p * np.log(V/V_ref)
        partials['m_empty', 'V_ref'] = -(p_base * delta_p) * m_wing / V_ref

        partials['m_empty', 'm_fuse'] = 1.0

class EngineWeightComp(om.ExplicitComponent):
    """
    Component for "EngineWeightComp" box containing analytical derivatives
    """
    def initialize(self):
        self.options.declare('vec_size', default=1, types=int)

    def setup(self):
        n = self.options['vec_size']

        #proposed design variables
        self.add_input('SFC_tech', units="unitless")

        #model variable (output from other component)
        #n/a

        #uncertain parameters
        self.add_input('delta_alpha', val=1.0, units='unitless', shape=(n,))
        
        #tuning parameters
        self.add_input('alpha_base', units='unitless')

        #constant parameters
        self.add_input('m_eng_ref', val=parameters['m_eng_ref'], units='kg')

        #outputs
        self.add_output('m_engine', units='kg', desc='Engine mass', shape=(n,))

    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        
        self.declare_partials('m_engine', ['SFC_tech', 'm_eng_ref', 'alpha_base'])
        
        self.declare_partials('m_engine', ['delta_alpha'], rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        """
        m_engine = m_eng_ref * (1 + alpha_base * delta_alpha * SFC_tech)
        """
        SFC_tech = inputs['SFC_tech']
        m_eng_ref = inputs['m_eng_ref']
        alpha_base = inputs['alpha_base']
        delta_alpha = inputs['delta_alpha']
        
        outputs['m_engine'] = m_eng_ref * (1 + alpha_base * delta_alpha * SFC_tech)
    
    def compute_partials(self, inputs, partials):
        m_eng_ref = inputs['m_eng_ref']
        alpha_base = inputs['alpha_base']
        SFC_tech = inputs['SFC_tech']
        delta_alpha = inputs['delta_alpha']
        
        partials['m_engine', 'SFC_tech'] = m_eng_ref * (alpha_base * delta_alpha)

        partials['m_engine', 'm_eng_ref'] = (1 + alpha_base * delta_alpha * SFC_tech)
        partials['m_engine', 'alpha_base'] = m_eng_ref * (delta_alpha * SFC_tech)

        partials['m_engine', 'delta_alpha'] = m_eng_ref * (alpha_base * SFC_tech)