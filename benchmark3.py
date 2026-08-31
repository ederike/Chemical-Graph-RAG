from benchmark3.config import DEFAULT_CONFIG_PATH as config_3
from benchmark3.workflow import Benchmark3Workflow

if __name__ == "__main__":
    wf = Benchmark3Workflow.from_config(config_3)
    wf.run()
