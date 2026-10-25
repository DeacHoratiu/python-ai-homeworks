import os
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

def list_files() -> str:
    """Returns all files at project level except .venv/, .git/, .idea/, .pytest_cache/, __pycache__/ folders"""
    try:
        result = subprocess.run(
            ["cmd","/c","dir","/s","/b"],
            cwd=project_root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60
        )

        output = result.stdout or ""
        #Remove unwanted lines from the output
        output_lines = output.splitlines()
        filtered_lines = []

        for line in output_lines:
            if ".venv" in line:
                continue
            if ".git" in line:
                continue
            if ".idea" in line:
                continue
            if "__pycache__" in line:
                continue
            if ".pytest_cache" in line:
                continue

            filtered_lines.append(line)

        return "\n".join(filtered_lines) or "No files could be found"
    except Exception as e:
        return f"error: {e}"

def search_keyword():
    """Searches for 'pytest' in project files"""
    try:
        result = subprocess.run(
            ["cmd","/c","findstr","/s","/i","/p","pytest","*.*"],
            cwd=project_root,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=60
            )

        output = result.stdout or ""
        output_lines = output.splitlines()
        filtered_lines = []

        # Remove unwanted lines from the output
        for line in output_lines:
            if ".venv" in line:
                continue
            if ".git" in line:
                continue
            if ".idea" in line:
                continue
            if "__pycache__" in line:
                continue
            if ".pytest_cache" in line:
                continue

            filtered_lines.append(line)

        return "\n".join(filtered_lines) or "No matches found"
    except Exception as e:
        return f"error: {e}"

def run_tool(client: OpenAI, tool_call):
    tool_name = tool_call.function.name

    if tool_name == "list_files":
        return list_files()

    if tool_name == "search_keyword":
        return search_keyword()

    return f"error: unknown tool {tool_name}"


def complete(client: OpenAI, messages: list[dict], tools: list[dict]):
    response = client.chat.completions.create(
        messages=messages,
        model= os.getenv("OPENROUTER_MODEL"),
        tools = tools,
        tool_choice = "auto"
    ).choices[0].message
    return response

def main():
    load_dotenv()

    client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
            )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Lists repository files, excluding .venv, .git, .idea and cache folders. Use this first before searching for pytest"
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_keyword",
                "description": "Searches for the keyword 'pytest' in repository files, excluding .venv, .git, .idea and cache folders. Use this after listing the files.",
            }
        }
    ]

    messages = []
    messages.append({
        "role": "system",
        "content": "You are a helpful repository search agent. If the user asks to search for pytest, inspect the project by first calling list_files, then call search_keyword. If the user only greets you or asks something unrelated to searching, answer normally without using tools .In the final answer after a search, show only the files or lines where 'pytest' was found. Do not invent files. Use only the tool outputs.",
    })
    while True:
        inp = input("you>")
        if inp == "quit":
            break
        messages.append({
            "role": "user",
            "content": inp
        })


        response = complete(client,messages,tools)
        messages.append(response.model_dump(exclude_none=True))

        while response.tool_calls:
            for tool_call in response.tool_calls:
                output = run_tool(client, tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })
            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))

        print("agent> ", response.model_dump_json(indent=2))
        print(f"Message count: {len(messages)}")

if __name__ == "__main__":
    try:
        print("Folositi comanda 'search pytest in the project' pentru a rula tool-urile")
        main()
        # print(search_keyword())
    except Exception as e:
        print(e)
