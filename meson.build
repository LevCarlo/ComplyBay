project(
  'complybay',
  'c', 'fortran',
  version: '0.2.0.dev0',
  license: 'MIT',
  meson_version: '>=1.1.0',
  default_options: ['buildtype=release', 'warning_level=1'],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(
  py, ['-c', 'import numpy; print(numpy.get_include())'],
  check: true,
).stdout().strip()
incdir_f2py = run_command(
  py, ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true,
).stdout().strip()
inc_f2py = include_directories(incdir_numpy, incdir_f2py)

subdir('src/complybay')
