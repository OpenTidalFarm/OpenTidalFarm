tests: mesh.xml
	@py.test .

mesh.xml:
	@echo "--- Making a mesh required for testing ---"
	@gmsh -2 mesh.geo > mesh.log
	@dolfin-convert -o xml mesh.msh mesh.xml >> mesh.log
	@../scripts/convert_to_new_xml.py >> mesh.log

clean:
	@rm -f mesh.msh mesh.xml mesh.log
