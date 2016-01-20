module add miniconda/2.7
conda create -p ~/Athena/env pip
source activate ~/Athena/env
conda install numpy libgfortran scipy pandas scikit-learn nltk sympy funcsigs cython ipython ipython-notebook matplotlib numexpr tornado accelerate dateutil
conda list > ~/Athena/requirements.txt
