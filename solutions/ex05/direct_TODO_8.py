    J_jac = sym.lambdify([z], sym.derive_by_array(J, z), modules='numpy') 