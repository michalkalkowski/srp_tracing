from ogilvy.ogilvy_model import Ogilvy_weld as o_weld
from mina.original import MINA_weld

a = 60
b = 4
c = 40


# Create a MINA model for the weld first
weld_parameters = dict([('T', 1),
                        ('n_weld', 1),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
oweld = o_weld(weld_parameters)
oweld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=1.)
oweld.solve()

weld_parameters = dict([('remelt_h', 0.3),
                        ('remelt_v', 0.12),
                        ('theta_b', np.deg2rad(18.5)),
                        ('theta_c', np.deg2rad(10)),
                        ('number_of_layers', 30),
                        ('number_of_passes', np.array([1]*4 + 5*[2] + 8*[3] +
                                                      [4]*9 + [5]*4,)),
                        ('electrode_diameter', np.array([2.4, 2.4, 2.4, 2.4] +
                                                        [5]*13 + [4]*13)),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
weld = MINA_weld(weld_parameters)
weld.define_order_of_passes('right_to_left')
weld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=1.)
weld.solve()
weld.plot_grain_orientations()
oweld.plot_grain_orientations()
weld_parameters = dict([('T', 1),
                        ('n_weld', 2),
                        ('a', a),
                        ('b', b),
                        ('c', c)])
oweld = o_weld(weld_parameters)
oweld.define_grid_size(2, use_centroids=True, add_boundary_cells=True,
                      boundary_offset=1.)
oweld.solve()
oweld.plot_grain_orientations()
multipl = np.copy(oweld.grain_orientations_full)
m2 = -1.4*multipl - 1.12
m2[multipl > -0.8] = 0
mmin = np.copy(m2)
mmin[m2 == 0] = 1
mmin[m2 != 0] = 0
total = m2*oweld.grain_orientations_full + mmin*weld.grain_orientations_full
grf = np.copy(total)
grains = grf[~np.isnan(total)]
weld.grain_orientations = grains
total = m2*oweld.grain_orientations_full + mmin*weld.grain_orientations_full
total[27, 0] = -1.4
