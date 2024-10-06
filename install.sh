#!/bin/bash
ENVNAME="BELKA"

is_installed=$(conda info --envs | grep $ENVNAME -c)

if [[ "${is_installed}" == "0" ]];then
  conda create -n $ENVNAME python=3.10 -y;
  conda activate $ENVNAME
  pip install -r requirements.yml
fi

if [[ `command -v activate` ]]
then
  source `which activate` $ENVNAME
else
  conda activate $ENVNAME
fi

# Check to make sure MIMS is activated
if [[ "${CONDA_DEFAULT_ENV}" != $ENVNAME ]]
then
  echo "Could not run conda activate $ENVNAME, please check the errors";
  exit 1;
fi

pip_exc="${CONDA_PREFIX}/bin/pip"

$pip_exc install -e . # For development

python -m pip install --upgrade build installer


pytorch_version=$(python <<EOF
import torch
print(f"torch-{torch.__version__}.html")
EOF
)
echo $pytorch_version
# # Use pytorch_version in pip install commands
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/$pytorch_version
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/$pytorch_version
pip install torch-geometric
pip install torchvision