import os
from app.core.logging import logger
from app.core.config import (
    Environment,
    settings,
)
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama


def get_llm(
    model: str,
    top_p: float = None,
    top_k: float = None,
    presence_penalty: float = None,
    frequency_penalty: float = None,
):
    """Retrieve the specified language model based on the model name."""
    model = model.lower().strip().replace("-", "_")
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)
    temperature = settings.DEFAULT_LLM_TEMPERATURE
    api_key = settings.LLM_API_KEY
    max_tokens = settings.MAX_TOKENS

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logger.error(err)
        raise Exception(err)

    logger.info("Model: {}".format(env_key))
    try:
        if "openai" in model:
            model_name, api_key = env_value.split(",")
            if "o3-mini" in model:
                llm = ChatOpenAI(api_key=api_key, model=model_name)
            else:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

        elif "anthropic" in model:
            model_name, api_key = env_value.split(",")
            llm = ChatAnthropic(api_key=api_key, model=model_name, temperature=temperature, timeout=None)

        elif "ollama" in model:
            model_name, base_url = env_value.split(",")
            llm = ChatOllama(base_url=base_url, model=model_name)

        else:
            model_name, api_endpoint, api_key = env_value.split(",")
            llm = ChatOpenAI(
                api_key=settings.api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logger.error(err)
        raise Exception(err)

    logger.info(f"Model created - Model Version: {model}")
    return llm, model_name
