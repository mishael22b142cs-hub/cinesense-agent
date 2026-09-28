


from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from step5_langgraph import graph          # 🔴 Step 5-ലെ അതേ flowchart (nodes + edges) വീണ്ടും ഉപയോഗിക്കുന്നു

memory = InMemorySaver()                   # 🔴 notebook ഉണ്ടാക്കുന്നു: conversations RAM-ൽ save ചെയ്യും
app = graph.compile(checkpointer=memory)   # 🔴 graph-ന് notebook കൊടുക്കുന്നു, ഇനി ഓരോ step-ഉം save ആകും


def chat(message, thread_id):
    config = {
        "configurable": {"thread_id": thread_id},   # 🔴 ഏത് customer-ന്റെ page തുറക്കണം എന്ന് പറയുന്നു
        "recursion_limit": 12,
    }
    inputs = {"messages": [HumanMessage(content=message)]}   # 🔴 പുതിയ message മാത്രം അയച്ചാൽ മതി; പഴയവ notebook-ൽ നിന്ന് തനിയെ വരും

    try:
        for update in app.stream(inputs, config, stream_mode="updates"):
            for node_name, data in update.items():
                if node_name == "agent":
                    msg = data["messages"][-1]
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"   🧠 uses {tc['name']}({tc['args']})")
                    else:
                        print(f"🤖 {msg.content}\n")
    except GraphRecursionError:
        print("⛔ Stopped: too many steps.\n")
    except Exception as e:
        print(f"⛔ Error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    # ---------- DEMO: memory പ്രവർത്തിക്കുന്നുണ്ടോ എന്ന് കാണാൻ ----------
    print("=" * 20, "Conversation 1 (thread: mishael)", "=" * 20)
    print("👤 I love Christopher Nolan movies. Keep your answers short.")
    chat("I love Christopher Nolan movies. Keep your answers short.", "mishael")

    print("👤 Suggest 3 movies for me.")
    chat("Suggest 3 movies for me.", "mishael")              # 🔴 "for me" = Nolan ഇഷ്ടമാണ് എന്ന് ഓർക്കണം

    print("👤 Which of those has the highest rating?")
    chat("Which of those has the highest rating?", "mishael")  # 🔴 "those" = തൊട്ടുമുമ്പ് suggest ചെയ്ത movies

    print("=" * 20, "Conversation 2 (thread: new_user)", "=" * 20)
    print("👤 Suggest 3 movies for me.")
    chat("Suggest 3 movies for me.", "new_user")             # 🔴 പുതിയ page: Nolan-നെ കുറിച്ച് ഒന്നും അറിയില്ല

    # ---------- നിങ്ങൾക്ക് നേരിട്ട് chat ചെയ്യാം ----------
    print("=" * 20, "Your turn! (type 'quit' to exit)", "=" * 20)
    while True:
        user_text = input("👤 You: ")
        if user_text.strip().lower() in ("quit", "exit"):
            break
        chat(user_text, "live_chat")