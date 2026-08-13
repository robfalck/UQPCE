import openmdao.api as om
import jax.numpy as jnp
import numpy as np

from fixed import parameters, tuning


class WeightsComp(om.JaxExplicitComponent):
    """
    Evaluates the weights & structures for a coupled Breguet range with MDAO & JAX
    
    Inputs:
    Design Vars: S [m^2], AR [-], V [m/s]
    Vector coupling inputs: m_total [kg], m_engine [kg]
    Vector uncertain inputs: delta_kw, delta_fsys, delta_p
    Fixed parameters: kw_base, fsys_base, p_base, V_ref [m/s], m_fuse [kg]

    Outputs:
    Vectors: m_empty [kg], m_wing [kg]
    """

    def initialize(self):
        self.options.declare("vec_size", types=int)

    def setup(self):
        n = self.options["vec_size"]

        self.add_input("S", units="m**2")
        self.add_input("AR", units="unitless")
        self.add_input("V_cruise", units="m/s")

        self.add_input("m_total", shape=(n,), units="kg")
        self.add_input("m_engine", shape=(n,), units="kg")

        self.add_input("delta_kw", val=jnp.ones(n), shape=(n,), units="unitless")
        self.add_input("delta_fsys", val=jnp.ones(n), shape=(n,), units="unitless")
        self.add_input("delta_p", val=jnp.ones(n), shape=(n,), units="unitless")

        self.add_input("kw_base", units="unitless")
        self.add_input("fsys_base", units="unitless")
        self.add_input("p_base", units="unitless")
        self.add_input("V_ref", val=parameters["V_ref"], units="m/s")
        self.add_input("m_fuse", val=parameters["m_fuse"], units="kg")

        # res_ref mirrors disciplines/weight.py: scales residuals to O(1) for the
        # AeroStruct Newton solve.
        self.add_output("m_empty", shape=(n,), units="kg", res_ref=1.0e4)
        self.add_output("m_wing", shape=(n,), units="kg", res_ref=1.0e3)

    def setup_partials(self):
        n = self.options['vec_size']
        arange = np.arange(n)
        # Vector inputs are elementwise across the sample dimension, so these
        # subjacs are diagonal. Without this OpenMDAO assumes dense (n x n),
        # which made the AeroStruct sparse LU ~82x more expensive.
        self.declare_partials(of='m_empty', wrt=['m_total', 'm_engine', 'delta_kw', 'delta_fsys', 'delta_p'], rows=arange, cols=arange)
        self.declare_partials(of='m_empty', wrt=['S', 'AR', 'V_cruise', 'kw_base', 'fsys_base', 'p_base', 'V_ref', 'm_fuse'])
        self.declare_partials(of='m_wing', wrt=['m_total', 'm_engine', 'delta_kw', 'delta_fsys', 'delta_p'], rows=arange, cols=arange)
        self.declare_partials(of='m_wing', wrt=['S', 'AR', 'V_cruise', 'kw_base', 'fsys_base', 'p_base', 'V_ref', 'm_fuse'])

    def compute_primal(
        self,
        S,
        AR,
        V_cruise,
        m_total,
        m_engine,
        delta_kw,
        delta_fsys,
        delta_p,
        kw_base,
        fsys_base,
        p_base,
        V_ref,
        m_fuse):
        
        m_wing = (kw_base * delta_kw * S**0.758 * AR**0.6 * m_total**0.006 * (V_cruise / V_ref) ** (p_base * delta_p))

        m_empty = (m_wing + m_fuse + fsys_base * m_total * delta_fsys + m_engine)

        return m_empty, m_wing