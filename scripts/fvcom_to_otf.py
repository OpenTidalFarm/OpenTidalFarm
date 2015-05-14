from pylab import *
import matplotlib.tri as Tri
import netCDF4
import datetime as dt


class FVCOMReader(object):

    def __init__(self, ncfile):
        self.nc = netCDF4.Dataset(ncfile).variables

    @property
    def triangles(self):
        return self.nc["trinodes"]

    @property
    def nodes(self):
        return zip(self.nc["x"], self.nc["y"])

    @property
    def velocity_x(self):
        for timelevel in range(len(self.nc["ua"])):
            yield self.nc["ua"][timelevel]

    @property
    def velocity_y(self):
        for timelevel in range(len(self.nc["va"])):
            yield self.nc["va"][timelevel]

    @property
    def velocity(self):
        for timelevel in range(len(self.nc["ua"])):
            yield zip(self.nc["ua"][timelevel], self.nc["va"][timelevel])

    @property
    def julianTime(self):
        return self.nc["julianTime"]


class FEniCSWriter(object):

    def write_mesh(self, nodes, triangles, filename):
        """ Writes a FEniCS compatible xml mesh.

            Parameters:

            nodes: A list of (x, y) tuples with the node positions
            triangles: A list of (n1, n2, n3) tuples with the node ids of each
                       triangle
            filename: The output file (.xml) """

        f = open(filename, "w")

        # Write header
        f.write('<?xml version="1.0"?>\n')
        f.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
        f.write('  <mesh celltype="triangle" dim="{}">\n'.format(2))

        # Write nodes
        f.write('    <vertices size="{}">\n'.format(len(nodes)))

        for i, (x, y) in enumerate(nodes):
            f.write('      <vertex index="{}" x="{}" y="{}" />\n'.format(i, x, y))
        f.write('    </vertices>')

        # Write elements
        f.write('    <cells size="{}">\n'.format(len(triangles)))

        for i, (n1, n2, n3) in enumerate(triangles):
            f.write('      <triangle index="{}" v0="{}" v1="{}" v2="{}" />\n'.format(i, n1, n2, n3))
        f.write('    </cells>\n')

        # Write footer
        f.write('  </mesh>\n')
        f.write('</dolfin>')

        f.close()

    def write_dg0_function(self, values, filename):
        """ Writes a FEniCS compatible DG0 function in xml format.

            Parameters:

            values: A list of float or tuple of floats with the function values
            filename: The output file (.xml) """


        f = open(filename, "w")

        # Write header
        f.write('<?xml version="1.0"?>\n')
        f.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')

        # Vector or scalar function?
        if not hasattr(values[0], '__iter__'):
            dim = 1
        else:
            dim = len(values[0])

        # Write function values
        f.write('  <function_data size="{}">\n'.format(len(values)*dim))
        idx = 0
        for i, value in enumerate(values):
            if dim == 1:
                f.write('    <dof index="{}" value="{}" cell_index="{}" cell_dof_index="{}" />\n'.format(idx, value, i, 0))
                idx += 1
            else:
                for d, v in enumerate(value):
                    f.write('    <dof index="{}" value="{}" cell_index="{}" cell_dof_index="{}" />\n'.format(idx, value[d], i, d))
                    idx += 1
        f.write('  </function_data>\n')

        # Write footer
        f.write('</dolfin>')

def plot(args):
    from dolfin import plot, interactive, Mesh, Function, VectorFunctionSpace

    # Plot the mesh
    mesh = Mesh(args.xml)
    plot(mesh, title="Mesh")
    interactive()

    # Plot velocities
    if args.velocity is not None:
        G = VectorFunctionSpace(mesh, "DG", 0)
        base_file = args.velocity[:-4] + "_{}.xml"

        for i, u in enumerate(fvcom_reader.velocity):
            g = Function(G, base_file.format(i))
            plot(g, title="Velocity time={}".format(i))
            interactive()


if __name__ == "__main__":
    import argparse

    # Read the command line arguments
    parser = argparse.ArgumentParser(description="Converts FVCOM meshes and velocity fields to OpenTidalFarm compatible xml files")
    parser.add_argument('--nc', required=True, help='input FVCOM filename (.nc extension)')
    parser.add_argument('--xml', required=True, help='output OpenTidalFarm mesh filename (.xml extension)')
    parser.add_argument('--velocity', help='output OpenTidalFarm velocity filename (.xml extension)')
    parser.add_argument('--plot', action='store_true', help='plot the results')
    args = parser.parse_args()

    fvcom_reader = FVCOMReader(args.nc)

    fenics_writer = FEniCSWriter()
    fenics_writer.write_mesh(fvcom_reader.nodes, fvcom_reader.triangles, args.xml)
    print "Wrote {}".format(args.xml)

    # Write velocity fields
    if args.velocity is not None:
        base_file = args.velocity[:-4] + "_{}.xml"
        for i, u in enumerate(fvcom_reader.velocity):
            fenics_writer.write_dg0_function(u, base_file.format(i))
            print "Wrote {}".format(base_file.format(i))

    print "Conversion finished."

    if args.plot:
        plot(args)
