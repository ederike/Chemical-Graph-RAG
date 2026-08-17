from benchmark2.config import DEFAULT_CONFIG_PATH
from benchmark2.workflow import Benchmark2Workflow

if __name__ == "__main__":
    wf = Benchmark2Workflow.from_config(DEFAULT_CONFIG_PATH)
    wf.run()
