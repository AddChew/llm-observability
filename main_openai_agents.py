import asyncio

from httpx import AsyncClient
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

from opentelemetry.sdk import trace as trace_sdk
from openinference.instrumentation import TraceConfig

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

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
OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

async def main():
    agent = Agent(
        name="History Tutor",
        instructions="You answer history questions clearly and concisely.",
        model=OpenAIChatCompletionsModel( 
            model="gpt-4.1",
            openai_client=AsyncOpenAI(
                api_key="api-token",
                http_client=AsyncClient(),
                base_url="http://localhost:4242/v1",
            )
        )
    )
    result = await Runner.run(agent, "When did the Roman Empire fall?")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())