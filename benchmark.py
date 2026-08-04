from benchmark.workflow import TestQueryWorkflow

if __name__ == "__main__":
    wf = TestQueryWorkflow.from_config("benchmark/config.yaml")
    # dataset = wf.generate_questions()
    # report = wf.evaluate()
    report = wf.run_all()
