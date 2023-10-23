import importlib
import os 
# import scipy_openblas64 as openblas

if __name__ == '__main__':
  # print({ 'lib_dir' : openblas.get_lib_dir() })
  basedir = os.getcwd()
  openblas_dir = os.path.join(basedir, ".openblas")
  pkg_config_fname = os.path.join(openblas_dir, "scipy-openblas.pc")
  blas_variant = '32'
  module_name = f"scipy_openblas{blas_variant}"
  try:
    openblas = importlib.import_module(module_name)
  except ModuleNotFoundError:
    raise RuntimeError(f"'pip install {module_name} first")
  os.makedirs(openblas_dir, exist_ok=True)
  pkg_config = openblas.get_pkg_config().split('\n')
  for i, config_str in enumerate(pkg_config):
    if config_str[:5].upper() == 'LIBS:' and len(config_str) <= 6:
      config_str += r"-L${libdir}"
      pkg_config[i] = config_str
  pkg_config = '\n'.join(pkg_config)
  with open(pkg_config_fname, "wt", encoding="utf8") as fid:
    fid.write(pkg_config.replace("\\", "/"))
  print(openblas_dir)