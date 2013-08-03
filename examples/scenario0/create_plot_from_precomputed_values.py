import matplotlib.pyplot as plt
import numpy

P = [0, 0.0, 2.5214346244052885, 3.0463826868929584, 3.1941651487312024, 3.2280373854290536, 3.219356060128948, 3.1931966659159134, 3.159827132105891, 3.123868757683319, 3.0875192650492282, 3.0518462224564136, 2.984277476534521, 2.8664448702856884, 2.768824336075173, 2.6866923428484037, 2.6163231650631786, 2.5550467623911843, 2.50095092853066, 2.4526406135187306]
f = [0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0, 200.0]

scaling = 0.7
plt.figure(1, figsize = (scaling*7., scaling*4.))
plt.gcf().subplots_adjust(bottom=0.15)
plt.plot(f, P, color = "black")
plt.ylabel('Power output [MW]')
plt.xlabel('Friction coefficient K')
plt.yticks(numpy.arange(0, 3.5, 1))
plt.savefig('turbine_friction_vs_power_precomputed.pdf')
