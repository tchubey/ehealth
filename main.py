import argparse
import warnings
from pathlib import Path
from scripts.utils import Collection

from scripts.submit import Run, handle_args

from mytools.JointModel import ConcreteModel
from mytools.misc import config_seeds
def main(tasks):
    if not tasks:
        warnings.warn("The run will have no effect since no tasks were given.")
        return
    config_seeds(seed = 0)
    
    name = "model_8"

    model = ConcreteModel(model_name = name, config_path="config/model_params.json")
    
    model.train(finput_train = Path("data/training/"), finput_valid = Path("data/development/main"))
   
    Run.submit(name, tasks, model)
    


if __name__ == "__main__":
    import logging
    logging.basicConfig(
    filename='logs/logfile.log',
    format='%(asctime)s  %(levelname)s %(module)-5s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().setLevel(logging.INFO)
    tasks = handle_args()
    main(tasks)
