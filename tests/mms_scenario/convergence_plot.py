from dolfin import MPI
import dolfin_adjoint
from matplotlib import rc  
import matplotlib.pyplot as plt

def save_convergence_plot(errors, element_sizes, title, legend, order, offset = 0.8, show_title = True, xlabel = 'Element size [m]', ylabel = r"$L_2$ error"):
    ''' Creates a convergence plot '''
    if MPI.process_number() != 0:
        return

    # Plot the errors
    scaling = 0.7
    rc('text', usetex=True)
    plt.figure(1, figsize=(scaling*7,scaling*4))

    plt.xlabel('Element size')
    plt.ylabel('Error')
    if show_title:
        plt.title(title)

    plt.loglog(element_sizes, errors, 'gx', label = legend, color = 'k')
    # Construct an error plot with the expected order
    c = errors[-1]/element_sizes[-1]**order*offset
    expected_errors = [c*element_size**order for element_size in element_sizes] 
    print order
    if order == 1.0:
        order_str = "First order"
    elif order == 2.0:
        order_str = "Second order"
    else:
        order_str = str(order) + " order"
    plt.loglog(element_sizes, expected_errors, label = order_str, color = 'k')

    plt.legend(loc='best')
    #plt.axis([min(element_sizes)*1.01, max(element_sizes)*1.01, min(errors)*1.01, max(errors)*1.01])
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    filename = '_'.join(title.split()).lower()
    print 'Saving as: ', filename 
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.pdf')
    plt.close()


if __name__ == '__main__':
    T = 24*60*60 
    errors = [208.0311618136566, 112.87714516555415, 58.046875548809496, 29.240855065130535] 
    dts = [1.8060945639429236, 0.9030472819714618, 0.4515236409857309, 0.22576182049286544]
    print "Error summary: ", errors
    print "Element sizes: ", dts
    save_convergence_plot(errors, dts, "Temporal rate of convergence test", "Temporal error", order = 1.0, show_title = False, xlabel = "Time step [s]")
    print "Convergence orders: ", dolfin_adjoint.convergence_order(errors[0:1] + errors[2:])
