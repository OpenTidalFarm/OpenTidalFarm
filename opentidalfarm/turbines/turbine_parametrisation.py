import numpy as np
from matplotlib import pyplot as plt


class TurbineParametrisation(object):

    def __init__(self, u_in=1, u_rated=2.5, ct_design= 0.6, diameter=20,
                 rho=1000):
        """ 
        u_in: cut in velocity
        u_rated: rated velocity
        ct_design: design thrust coefficient
        diameter: turbine diameter
        rho: density of medium (water)
        """
        self.u_in = u_in
        self.u_rated = u_rated
        self.ct_design = ct_design
        self.diameter = diameter
        self.rho = rho

        self.swept_area = np.pi * (diameter/2)**2

    def thrust_coefficient(self, u):
        """ calculate the thrust coefficient

        u: turbine's upstream velocity
        """
        if u <= u_in:
            return 0
        elif u_in < u and u < u_rated:
            return ct_design
        else:
            return (ct_design * u_rated**3) / u**3
    
    def power_coefficient(self, u):
        """ calculate the power coefficient

        u: turbine's upstream velocity
        """
        C_t = self.thrust_coefficient(u)
        return 0.5*C_t*(1+np.sqrt(1-C_t))

    def power(self, u):
        """ calculate the power at a given velocity

        u: the turbine's upstream velocity
        """
        C_p = self.power_coefficient(u)
        return 0.5 * self.rho * C_p * self.swept_area * u**3

    def plot_thrust_curve(self, u_min=0, u_max=6, save_fig=False):
        """ draw a plot of the thrust curve

        u_min: minimum velocity to include on the plot
        u_max: maximum velocity to include on the plot
        """
        speed = np.linspace(u_min, u_max, (u_max-u_min)*100)
        C_t = np.zeros(len(speed))
        for i in range(len(speed)):
            C_t[i] = self.thrust_coefficient(speed[i])
        plt.plot(speed, C_t)
        plt.title('Thrust Curve')
        plt.ylim([0, max(C_t)*1.05])
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Thrust Coefficient, $C_t$')
        if save_fig:
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif') 
            fig = plt.gcf()
            fig.set_size_inches(15, 12)
            fig.savefig('thrust_curve.png')
            fig.clf()
        else:
            plt.show()
            plt.clf()

    def plot_power_curve(self,  u_min=0, u_max=6, save_fig=False):
        """ draw a plot of the power curve

        u_min: minimum velocity to include on the plot
        u_max: maximum velocity to include on the plot
        """
        speed = np.linspace(u_min, u_max, (u_max-u_min)*100)
        power = np.zeros(len(speed))
        for i in range(len(speed)):
            power[i] = self.power(speed[i])
        plt.plot(speed, power)
        plt.title('Power Curve')
        plt.ylim([0, max(power)*1.05])
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Power (W)')
        if save_fig:        
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif') 
            fig = plt.gcf()
            fig.set_size_inches(15, 12)
            fig.savefig('power_curve.png')
            fig.clf()
        else:
            plt.show()
            plt.clf()



u_in = 1.
u_rated = 2.5  
diameter = 18.
ct_design = 0.6
rho = 1000

tp = TurbineParametrisation(u_in=u_in, 
                            u_rated=u_rated, 
                            ct_design=ct_design,
                            diameter=diameter, 
                            rho=rho)

print tp.power(2.5)

#tp.plot_power_curve(save_fig=False)


u = u_rated
