#! /usr/bin/env bash

: ${PROJECT:="kp-story-generation"}
VIRTUALENV_DIR=".virtualenv"

# If we already have a virtual environment activated,
# bail out and advise the user to deactivate.
OLD_VIRTUAL_ENV="${VIRTUAL_ENV}"
if [ "${OLD_VIRTUAL_ENV}" != "" ]; then
  echo "************************************************************************"
  echo "Please deactivate your current virtual environment in order to continue!"
  echo "$ deactivate"
  echo "************************************************************************"
  exit 1
fi

rm -rf "${VIRTUALENV_DIR}"
python3.6 -m venv --prompt ${PROJECT} "${VIRTUALENV_DIR}"
source "${VIRTUALENV_DIR}/bin/activate"
python -m pip install -U pip
python -m pip install -r requirements.txt

# Print some info about the sucess of the installation.
echo ""
echo "Setup complete!"
echo ""
echo "To begin working, simply activate your virtual"
echo "environment and deactivate it when you are done."
echo ""
echo "    $ source activate"
echo "    $ python ..."
echo "    $ deactivate"
echo ""

