from pydantic import BaseModel, Field, BaseSettings
from mlutil.text import rwkv_utils
from mlutil.text.rwkv_utils import RWKVGenerationArgs as GenerationArgs
from fastapi import FastAPI
from pathlib import Path as P
from typing import List


class ModelConfig(BaseModel):
    models_dir: str
    model_name: str
    tokenizer_name: str = Field(default="20B_tokenizer.json")

    def get_model_path(self):
        return P(self.models_dir).expanduser() / self.model_name


rwkv_utils.RWKVGenerationArgs


class GenerateRequest(BaseModel):
    context: str
    token_count: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: float = 50
    alpha_frequency: float = 0.1
    alpha_presence: float = 0.1
    token_ban: List[str] = []
    token_stop: List[str] = []


def load_config(config_path: str):
    return ModelConfig.parse_file(config_path)


def load_model(model_settings: ModelConfig):
    return rwkv_utils.RWKVPipelineWrapper.load(model_settings.get_model_path())


config = load_config("model_config.json")
model = load_model(config)
app = FastAPI()


@app.post("/generate_with_model")
def generate_with_model(
    text: str, max_tokens: int = 20, generation_args: GenerationArgs = GenerationArgs()
):
    generated_text = model.generate(
        context=text, token_count=max_tokens, args=generation_args
    )
    return {"generated_text": generated_text}
