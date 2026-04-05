import asyncio
import contextlib
import json
import requests
from openai import AsyncOpenAI
import sys

async def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "

def get_user_input() -> str:
    """Get input from user on the main thread so Ctrl+C is handled cleanly."""
    return input("You: ")

async def fetch_json(url: str, params: dict) -> dict:
    try:
        headers = {
            "X-API-Key": config["elyos_api_key"]
        }
        response = await asyncio.to_thread(
            requests.get,
            url,
            params=params,
            headers=headers,
            timeout=config["timeout"],
        )
        success = False
        response.raise_for_status()
        try:
            data = response.json()
            if len(data.keys()) > 0: # If the data returned from the API is empty
                success = True
        except ValueError:
            data = response.text
            if len(data) > 0: # If the data returned from the API is empty
                success = True
        return {
            "success": success,
            "data": data,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

async def get_weather(location: str) -> dict:
    """Fetch weather from API (~200ms)."""
    return await fetch_json(config["weather_url"], {"location": location})

async def research_topic(topic: str) -> dict:
    """Research a topic (3-8 seconds). Should be cancellable."""
    return await fetch_json(config["research_url"], {"topic": topic})

async def call_llm(user_input: str, conversation_history: list):
    """Send input to LLM, handle tool calls, yield streaming response."""
    conversation_history.append(f"User: {user_input}")
    
    with open(config["system_prompt_file_path"], "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(config["user_prompt_file_path"], "r", encoding="utf-8") as f:
        user_prompt_template = f.read().strip()
        user_prompt = user_prompt_template.format(
            user_utterance = user_input,
            conversation_history = conversation_history
        )
    
    input_list = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_prompt
                }
            ]
        },
    ]

    with open(config["tools_file_path"], "r", encoding="utf-8") as f:
        tools = json.load(f)

    # Providing the tools to the LLM API to get the correct tool usage
    response = await client.responses.create(
        model="gpt-5",
        tools=tools,
        input=input_list
    )

    input_list.extend(response.output)
    
    is_function_call_detected = False
    for item in response.output:
        if item.type == "function_call":
            is_function_call_detected = True
            
            if item.name == "get_weather":
                location = json.loads(item.arguments)["location"]
                tool_result = await get_weather(location)
            elif item.name == "research_topic":
                topic = json.loads(item.arguments)["topic"]
                tool_result = await research_topic(topic)
            
            if not tool_result["success"]:
                error_message = f"{item.name} tool faced some issues"
                conversation_history.append(error_message)
                async for word in stream_text(error_message):
                    yield word
                return
            
            input_list.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": json.dumps(tool_result)
            })

    # Handling cases where the LLM is not able to get the appropriate tool to be used
    if not is_function_call_detected:
        error_message = "Please say an utterance that is supported by get_weather or research_topic tool"
        conversation_history.append(error_message)
        async for word in stream_text(error_message):
            yield word
        return

    full_output = ""
    async with client.responses.stream(
        model="gpt-5",
        instructions=system_prompt,
        input=input_list
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                full_output += event.delta
                yield event.delta

    conversation_history.append(f"Assistant: {full_output}")
    
async def spinner(message: str, stop_event: asyncio.Event):
    symbols = "|/-\\"
    i = 0
    
    try:
        while not stop_event.is_set():
            sys.stdout.write(f"\r{message} {symbols[i % len(symbols)]}")
            sys.stdout.flush()
            i += 1
            await asyncio.sleep(config["delay"])
    except asyncio.CancelledError:
        pass

    sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
    sys.stdout.flush()

async def execute_user_query(user_input: str, conversation_history: list):
    stop_event = asyncio.Event()
    spinner_task = asyncio.create_task(spinner("Thinking...", stop_event))
    history_len_before = len(conversation_history)
    try:
        first_chunk = True
        async for chunk in call_llm(user_input, conversation_history):
            # Stopping the spinner when we get the first chunk
            if first_chunk:
                stop_event.set()
                await spinner_task
                first_chunk = False
            print(chunk, end='', flush=True)
            await asyncio.sleep(config["delay"])
        print()

    except asyncio.CancelledError:
        del conversation_history[history_len_before:]
        print()
        raise

    finally:
        stop_event.set()
        with contextlib.suppress(asyncio.CancelledError):
            await spinner_task

async def run_single_query(user_input, conversation_history):
    await execute_user_query(user_input, conversation_history)

def main():
    conversation_history = []

    while True:
        try:
            user_input = get_user_input()
        except KeyboardInterrupt:
            print("\nInterrupted. Ready for new input.")
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        try:
            asyncio.run(run_single_query(user_input, conversation_history))
        except KeyboardInterrupt:
            print("\nCancelling current request...")
            print("Ready for new input.")
            continue

if __name__ == "__main__":
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    client = AsyncOpenAI(api_key=config["openai_api_key"])
    main()
