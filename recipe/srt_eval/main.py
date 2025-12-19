import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_package_available
from lighteval.models.model_input import GenerationParameters


if is_package_available("accelerate"):
    from datetime import timedelta
    from accelerate import Accelerator, InitProcessGroupKwargs
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None

def main(model_name,task):

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
        use_wandb=True,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        custom_tasks_directory=None,  # Set to path if using custom tasks
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        dtype="float16",
        generation_parameters=GenerationParameters(
            temperature=0.6,
        ),
    )

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

if __name__ == "__main__":
    import sys
    model_name = sys.argv[1]
    task = sys.argv[2]

    print(f"Evaluating model: {model_name} , on task: {task}")
    main(model_name=model_name, task=task)