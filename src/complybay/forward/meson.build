raydep_wrappers = custom_target(
  'raydep_ft_wrappers',
  input: 'raydep_ft.f95',
  output: ['raydep_ftmodule.c', 'raydep_ft-f2pywrappers.f'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'raydep_ft', '--lower', '--build-dir', '@OUTDIR@'],
)

py.extension_module(
  'raydep_ft',
  ['raydep_ft.f95', raydep_wrappers, incdir_f2py / 'fortranobject.c'],
  include_directories: inc_f2py,
  dependencies: py_dep,
  install: true,
  subdir: 'complybay/forward',
)

py.install_sources(
  '__init__.py',
  'complykernel.py',
  'k_solver.py',
  'ncomp_fortran.py',
  'ncomp_matlab_py.py',
  subdir: 'complybay/forward',
)
