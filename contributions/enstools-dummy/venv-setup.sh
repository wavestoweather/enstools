#!/bin/bash
set -e
# create virtual environments with all dependencies

# site-specific setting can be made in environment-site-com.sh
source venv-functions.sh
setup_environment

# create a new environment if not yet done
if [[ ! -d venv ]] ; then
    python3 -m venv --prompt ${PACKAGE_NAME} venv
fi

# activate the new environement
source venv/bin/activate

# install all requirements
pip install --upgrade pip
pip install ipykernel numpy wheel
pip install -e git+https://gitlab.physik.uni-muenchen.de/w2w/enstools.git@master#egg=enstools

# install jupyter kernel
ipython kernel install --user --name enstools-${PACKAGE_NAME}

# override settings to use the venv-kernel.sh script
cat > ${HOME}/.local/share/jupyter/kernels/enstools-${PACKAGE_NAME}/kernel.json << EOF
{
 "argv": [
  "${PWD}/venv-kernel.sh",
  "{connection_file}"
 ],
 "display_name": "enstools-${PACKAGE_NAME}",
 "language": "python"
}
EOF

# install the nda-package editable into the environment
pip install -e .
