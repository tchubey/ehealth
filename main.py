import argparse
import warnings
from pathlib import Path
from scripts.utils import Collection

from scripts.submit import Run, handle_args
""" from scripts.baseline import Baseline

from mytools.model_spacy import Spacy
#from mytools.model_test import BERT_model
from mytools.model_last_try import BERT_model """
from mytools.ConcreteModel_23 import ConcreteModel

def main(tasks):
    if not tasks:
        warnings.warn("The run will have no effect since no tasks were given.")
        return

    #baseline = Baseline()
    #baseline.train(Path("data/training/"))
    #Run.submit("baseline_1", tasks, baseline)    
    # spacy = Spacy()
    # spacy.train(Path("data/training/"), output_dir = "models/model_spacy/")    
    # Run.submit("Tania", tasks, spacy)
    
    name = "bert_re_23"
    bert = ConcreteModel(model_name = name)
    #bert.build_model()
    #bert.train(finput_train = Path("data/example"), finput_valid = Path("data/ex/"))
    #bert.train(finput_train = Path("data/training/"), finput_valid = Path("data/development/main"))
    finput = Path("data/example/")
    collection2 = (
        Collection().load_dir(finput)
        if finput.is_dir()
        else Collection().load(finput)
    )
    bert.run(collection2, taskA=True, taskB=False)
        
    #Run.submit(name, tasks, bert)
    

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
