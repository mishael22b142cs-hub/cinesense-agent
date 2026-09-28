from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage   # ✅ CHANGED: ToolMessage ചേർത്തു
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition                                # ✅ CHANGED: ToolNode ഒഴിവാക്കി
from langgraph.errors import GraphRecursionError

from step4_langchain import TOOLS, TOOLS_BY_NAME, llm_with_tools, SYSTEM_PROMPT   # ✅ CHANGED: TOOLS_BY_NAME ചേർത്തു; n_jobs line ഒഴിവാക്കി


# ================= NODE 1: agent (ചേട്ടൻ) =================
def agent_node(state: MessagesState):   # 🔴 ഇതുവരെയുള്ള messages (state) വാങ്ങി LLM-നെ വിളിക്കുന്നു
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}     # 🔴 LLM-ന്റെ മറുപടി state-ലെ notebook-ൽ ചേർക്കുന്നു


# ================= NODE 2: tools (നമ്മുടെ സ്വന്തം അടുക്കള) =================
def tools_node(state: MessagesState):                                     # ✅ CHANGED: പുതിയ function
    last = state["messages"][-1]                                          # ✅ CHANGED 🔴 agent അവസാനം എഴുതിയ order slip എടുക്കുന്നു
    results = []                                                          # ✅ CHANGED
    for tc in last.tool_calls:                                            # ✅ CHANGED 🔴 ഓരോ tool request-ഉം ഓരോന്നായി run ചെയ്യുന്നു (Step 4-ലെ അതേ രീതി)
        try:                                                              # ✅ CHANGED
            output = TOOLS_BY_NAME[tc["name"]].invoke(tc["args"])         # ✅ CHANGED
        except Exception as e:                                            # ✅ CHANGED
            output = f"Error: {type(e).__name__}: {e}"                    # ✅ CHANGED 🔴 error-ന്റെ type-ഉം കാണിക്കുന്നു, ശൂന്യമാകില്ല
        results.append(ToolMessage(content=str(output),                   # ✅ CHANGED
                                   tool_call_id=tc["id"], name=tc["name"]))  # ✅ CHANGED
    return {"messages": results}                                          # ✅ CHANGED 🔴 tool results state-ൽ ചേർക്കുന്നു


# ================= GRAPH (ചുവരിലെ board) =================
graph = StateGraph(MessagesState)                    # 🔴 messages list ആണ് state എന്ന് പറഞ്ഞ് ഒരു പുതിയ graph തുടങ്ങുന്നു
graph.add_node("agent", agent_node)                  # 🔴 പെട്ടി 1: LLM
graph.add_node("tools", tools_node)                  # ✅ CHANGED 🔴 പെട്ടി 2: നമ്മുടെ സ്വന്തം tools node
graph.add_edge(START, "agent")                       # 🔴 തുടക്കം എപ്പോഴും agent-ൽ
graph.add_conditional_edges("agent", tools_condition)  # 🔴 tool_calls ഉണ്ടെങ്കിൽ → "tools", ഇല്ലെങ്കിൽ → END
graph.add_edge("tools", "agent")                     # 🔴 tool result-മായി തിരിച്ച് agent-ലേക്ക്

app = graph.compile()                                # 🔴 board പൂർത്തിയായി, ഇനി run ചെയ്യാം


# ================= RUN =================
def run_agent(question):
    print(f"\n👤 USER: {question}")
    inputs = {"messages": [HumanMessage(content=question)]}
    config = {"recursion_limit": 12}                 # 🔴 ഇത്രയും steps കഴിഞ്ഞാൽ നിർത്തും (പഴയ max_steps)

    try:
        for update in app.stream(inputs, config, stream_mode="updates"):   # 🔴 ഓരോ node run ആകുമ്പോഴും അതിന്റെ ഫലം ഓരോന്നായി തരുന്നു
            for node_name, data in update.items():
                if node_name == "agent":
                    msg = data["messages"][-1]
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"🧠 agent → wants {tc['name']}({tc['args']})")
                    else:
                        print(f"🤖 ANSWER: {msg.content}")
                elif node_name == "tools":
                    for m in data["messages"]:
                        text = str(m.content)
                        if text.startswith("Error"):
                            print(f"❌ tools → {m.name} FAILED: {text}")
                        else:
                            print(f"🍳 tools → {m.name}: {text[:150]}...")
    except GraphRecursionError:
        print("⛔ Stopped: too many steps.")
    except Exception as e:                                                   # ✅ CHANGED
        print(f"⛔ Stopped because of an error: {type(e).__name__}: {e}")    # ✅ CHANGED 🔴 Groq error വന്നാലും program crash ആകില്ല


if __name__ == "__main__":
    run_agent("Recommend 3 movies like The Dark Knight.")                    # ✅ CHANGED: പുതിയ node test ചെയ്യാൻ വീണ്ടും ചേർത്തു
    run_agent("Tell me about Avatar and predict the rating of a similar 150 minute sci-fi movie with a 200 million dollar budget.")