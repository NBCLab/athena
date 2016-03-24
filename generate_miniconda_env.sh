module add miniconda/2.7
conda create -p /home/data/nbc/athena/athena/env pip
source activate /home/data/nbc/athena/athena/env
conda install numpy libgfortran scipy pandas scikit-learn nltk sympy funcsigs cython ipython ipython-notebook matplotlib numexpr tornado accelerate dateutil
conda list > /home/data/nbc/athena/athena/requirements.txt
