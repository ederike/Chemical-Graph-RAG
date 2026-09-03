from useless.benchmark2.config import DEFAULT_CONFIG_PATH as config_2
from useless.benchmark2.workflow import Benchmark2Workflow
    
if __name__ == "__main__":
    wf = Benchmark2Workflow.from_config(config_2)
    wf.run()


