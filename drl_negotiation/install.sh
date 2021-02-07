ROOT_PWD=$(pwd)/drl_negotiation
sudo apt install libopenmpi-dev, libgl1-mesa-glx
pip install -r $ROOT_PWD/requirement
pip install git+https://github.com/yasserfarouk/scml
pip install git+https://github.com/yasserfarouk/negmas
export PYTHONPATH=$PYTHONPATH:$ROOT_PWD
