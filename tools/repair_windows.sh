set -xe

WHEELDIR=$GITHUB_WORKSPACE/wheelhouse
WHEELNAME=$(find "$WHEELDIR" -type f | grep *primate*.whl)

# strip .local/bin/*.dll

# Make sure to leave the wheel in the same directory
# wheeldir=$(dirname $WHEELNAME)
# pushd $wheeldir
#   # Unpack the wheel and strip any .pyd DLLs inside
#   wheel unpack $WHEELNAME
#   rm $WHEELNAME
#   strip python_flint-*/flint/*.pyd
#   wheel pack python_flint-*
# popd

# Make the wheel relocatable. This will fail with an error message about
# --no-mangle if strip has not been applied to all mingw64-created .dll and
# .pyd files that are needed for the wheel.
delvewheel repair $WHEELNAME 	\
	--wheel-dir $WHEELHOUSE

# ## Need delvewheel for this 
# pip install delvewheel

# # create a temporary directory in the destination folder and unpack the wheel into there
# pushd $DEST_DIR
# mkdir -p tmp
# pushd tmp
# wheel unpack $WHEEL
# pushd primate*

# # To avoid DLL hell, the file name of libopenblas that's being vendored with
# # the wheel has to be name-mangled. delvewheel is unable to name-mangle PYD
# # containing extra data at the end of the binary, which frequently occurs when
# # building with mingw.
# # We therefore find each PYD in the directory structure and strip them.
# for f in $(find ./primate* -name '*.pyd'); do strip $f; done

# # now repack the wheel and overwrite the original
# wheel pack .
# mv -fv *.whl $WHEEL

# cd $DEST_DIR
# rm -rf tmp

# # the libopenblas.dll is placed into this directory in the cibw_before_build script.
# delvewheel repair -v -w $DEST_DIR $WHEEL