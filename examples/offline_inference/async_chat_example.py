"""An example of how to use the chat method with AsyncLLMEngine."""

import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams


async def main():
    # Create an AsyncLLMEngine from its config
    # This is the same as the LLM class, but for async
    engine_args = AsyncEngineArgs(model="facebook/opt-125m")
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Define the conversation
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        },
    ]

    # Generate a response
    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=128)
    request_id = "chat-0"
    results_generator = await engine.chat(messages=messages,
                                          sampling_params=sampling_params,
                                          request_id=request_id)

    # Print the results
    final_output = None
    async for request_output in results_generator:
        print(request_output.outputs[0].text)
        final_output = request_output

    # Ensure the final output is not None
    assert final_output is not None


if __name__ == "__main__":
    asyncio.run(main())
