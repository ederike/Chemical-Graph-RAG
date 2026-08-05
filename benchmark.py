from benchmark.config import DEFAULT_CONFIG_PATH
from benchmark.workflow import TestQueryWorkflow

if __name__ == "__main__":
    wf = TestQueryWorkflow.from_config(DEFAULT_CONFIG_PATH)
    wf.run()
