import argparse
import warnings
from pathlib import Path
from scripts.utils import Collection

from scripts.submit import Run, handle_args

from mytools.ConcreteModel_25 import ConcreteModel
from mytools.misc import config_seeds
def main(tasks):
    if not tasks:
        warnings.warn("The run will have no effect since no tasks were given.")
        return
    config_seeds()
    
    name = "bert_test_edit"
    model = ConcreteModel(model_name = name)
    
    model.train(finput_train = Path("data/example"))#, output_name='bert_apartideotromodelo', from_scratch = False)#, finput_valid = Path("data/ex/"))
    #model.train(finput_train = Path("data/training/"), finput_valid = Path("data/development/main"))
    
    finput = Path("data/example2/")
    collection2 = (
        Collection().load_dir(finput)
        if finput.is_dir()
        else Collection().load(finput)
    )
    
    model.run(collection2, taskA=True, taskB=True)
        
    #Run.submit(name, tasks, model)
    

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
