from benchmark.workflow import TestQueryWorkflow

if __name__ == "__main__":
    wf = TestQueryWorkflow.from_config("benchmark/config.yaml")
    # dataset = wf.generate_questions()   # 只生成并落盘
    report = wf.evaluate()               # 只评测：自动加载已有问题集
    # report = wf.run_all()              # 生成 + 评测一条龙
