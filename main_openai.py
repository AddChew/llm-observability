import asyncio

from httpx import AsyncClient
from openai import AsyncOpenAI

from opentelemetry.sdk import trace as trace_sdk
from openinference.instrumentation import TraceConfig
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from openinference.instrumentation.openai import OpenAIInstrumentor

config = TraceConfig(
    hide_inputs=True,
    hide_outputs=True,
    # hide_input_messages=hide_input_messages,
    # hide_output_messages=hide_output_messages,
    # hide_input_images=hide_input_images,
    # hide_input_text=hide_input_text,
    # hide_output_text=hide_output_text,
    # base64_image_max_length=base64_image_max_length,
)
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)


async def main():
    client = AsyncOpenAI(
        api_key="api-token",
        http_client=AsyncClient(),
        base_url="http://localhost:4242/v1"
    )
    response = await client.responses.create(
        input="Are you able to understand chinese?",
        model="gpt-4.1"
    )
    print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())