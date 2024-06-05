import Preparators.DataPreparator as DataPreparator
import Preparators.OptimizerPreparator as OptimizerPreparator
import Preparators.ModelPreparator as ModelPreparator
import Preparators.Iterator as Iterator
import Preparators.Asserter as Asserter


class Trainer:
    def __init__(self, config: dict):
        print("DEBUG: Initialization started")
        Asserter(config)
        print("DEBUG: No assertion errors found")
        self.dataset = DataPreparator.prepare(config["dataset"])
        print("DEBUG: Dataset prepared")
        self.optimizer = OptimizerPreparator.prepare(config["optimizer"])
        print("DEBUG: Optimizer prepared")
        self.model = ModelPreparator.prepare(config)
        print("DEBUG: Model prepared")
        self.iterator = Iterator(config)
        print("DEBUG: Iterator prepared")
        print("DEBUG: Initialization complete")

    def iterate(self) -> dict:
        return self.iterator.iterate(self.model, self.dataset)

    def finalize(self) -> dict:
        print("DEBUG: Finalization started")
        return self.iterator.finalize(self.model, self.dataset)
        print("DEBUG: Finalization complete")
