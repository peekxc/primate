
> python -m pip install --no-build-isolation --editable .

<!-- pythran --config compiler.blas=pythran-openblas src/primate2/lanczos.pycd -->
<!-- PYTHRANRC=../../.pythranrc pythran lanczos.py -->
<!-- python -m build . --no-isolation --wheel -->
<!-- PYTHRANRC=.pythranrc pythran src/primate2/_lanczos_pythran.py  -->


<!-- MACOSX_DEPLOYMENT_TARGET=14.0 pipx run cibuildwheel -->