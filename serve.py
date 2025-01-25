import os
from typing import Dict, Optional, List, Tuple, Union, Any, Tuple, Union, Any
import logging

from fastapi import FastAPI, Depends
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser
from functools import wraps
from starlette.middleware.cors import CORSMiddleware

# Configure logging
logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
            self,
            engine_args: AsyncEngineArgs,
            response_role: str,
            lora_modules: Optional[List[LoRAModulePath]] = None,
            chat_template: Optional[str] = None,
            model_name: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_name = model_name

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
            self,
            request: ChatCompletionRequest,
            raw_request: Request,
    ):
        """OpenAI-compatible HTTP endpoint"""
        try:
            if not self.openai_serving_chat:
                model_config = await self.engine.get_model_config()
                # Determine the name of the served model for the OpenAI client.
                if self.engine_args.served_model_name is not None:
                    served_model_names = self.engine_args.served_model_name
                else:
                    if self.model_name == None:
                        self.model_name = self.engine_args.model
                    served_model_names = [BaseModelPath(self.model_name, self.engine_args.model)]
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    base_model_paths=served_model_names,
                    response_role=self.response_role,
                    lora_modules=self.lora_modules,
                    chat_template=self.chat_template,
                    chat_template_content_format="openai",
                    prompt_adapters=None,
                    request_logger=None,
                )
            logger.info("Processing chat completion request")
            logger.debug(f"Request: {request}")

            generator = await self.openai_serving_chat.create_chat_completion(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                logger.error(f"Error response: {generator}")
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.code
                )

            if request.stream:
                logger.debug("Returning streaming response")
                return StreamingResponse(content=generator, media_type="text/event-stream")
            else:
                assert isinstance(generator, ChatCompletionResponse)
                logger.debug("Returning JSON response")
                return JSONResponse(content=generator.model_dump())

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": str(e)},
                status_code=500
            )


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs."""
    exclude_args_from_engine = ["model_name"]
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        if key in exclude_args_from_engine:
            continue
        if isinstance(value, list):
            for v in value:
                arg_strings.extend([f"--{key}", str(v)])
        elif isinstance(value, bool):
            if value:
                arg_strings.extend([f"--{key}"])
        else:
            arg_strings.extend([f"--{key}", str(value)])
    logger.info(f"Parsing CLI args: {arg_strings}")
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args



def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments."""
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    logger.info("Building Serve application")
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
        cli_args["model_name"]
    )


model = build_app(
    {
        "model": os.environ['MODEL_ID'],
        "model_name": os.environ.get('MODEL_NAME', None),
        "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
        "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM'],
        "gpu_memory_utilization": 1
    }
)
        



