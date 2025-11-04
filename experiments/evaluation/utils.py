import logging
from rich.logging import RichHandler
from datetime import datetime

from vllm import LLM


def setup_logging(loglevel: str = "INFO"):
    level = getattr(logging, loglevel.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logging.info(f"Logging initialized with level {loglevel}")


def generate_path(e_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"../output/{e_name}/"


def construct_llm(llm_config):
    llm_args = llm_config["llm"]
    sampling_args = llm_config["sampling"]
    llm = LLM(
        model=llm_args["model_name"],
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.5,
    )

    return llm