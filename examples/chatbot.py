"""
Simple personal assistant chatbot using roux
Run with: python examples/chatbot.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from roux import Memory

load_dotenv()

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
  """Calculate API cost"""
  costs = {
      'gpt-4o-mini': {
          'input':0.150 / 1_000_000,
          'output': 0.600 / 1_000_000,
      },
      'gpt-40': {
                   'input':2.50 / 1_000_000,
          'output': 10.00 / 1_000_000,
      }
  }

  rates = costs.get(model, costs['gpt-4o-mini'])
  return (input_tokens * rates['input']) + (output_tokens * rates['output'])

def main():
  # initialize
  client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
  memory = Memory()

  #Create session for user
  user_id = 'james'
  session = memory.create_session(participants=[user_id, 'assistant'])

  # System Prompt for personal assistant
  system_prompt = """
      You are a helpful personal assistant.
      Help the user remember tasks, follow-ups, and important information.
      Be concise and actionable. Track things they need to follow up on.
      """

  session.add_message('system', system_prompt, role='system')

  print("Personal Assistant (type 'quit' to exit)")
  print("-" * 50)

  total_cost = 0.0

  while True:

    # Get user input
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
      print(f"\nTotal session cost: ${total_cost:.4f}")
      print("Goodbye!")
      break

    if not user_input:
      continue

    # Add to memory
    session.add_message(user_id, user_input, role='user')

    # Get context and call LLM
    context = session.get_messages_for_llm()

    try:
      stream = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=context,
          temperature=0.7,
          max_tokens=500,
          stream=True,
          stream_options={"include_usage":True}
      )

      print("\nAssistant: ", end='', flush=True)

      # Collect the full response as it streams
      full_response = ""
      usage_info = None

      for chunk in stream:
        # Handle content chunks
        if chunk.choices and len(chunk.choices) > 0:
          delta = chunk.choices[0].delta
          if delta.content:
            content = delta.content
            print(content, end='', flush=True)
            full_response += content

        # Handle usage info (comes at the end)
        if hasattr(chunk, 'usage') and chunk.usage:
          usage_info = chunk.usage

      print()

      session.add_message('assistant', full_response, role='assistant')


      # Store assistant response
      if usage_info:

        cost = calculate_cost('gpt-4o-mini', usage_info.prompt_tokens, usage_info.completion_tokens)
        total_cost += cost
        print(f"\n[Tokens: {usage_info.total_tokens} | Cost: ${cost:.4f} | Total: ${total_cost:.4f}]")
    except Exception as e:
      print(f"\nError: {e}")


if __name__ == "__main__":
  main()

