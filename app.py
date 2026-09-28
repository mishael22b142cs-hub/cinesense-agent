import uuid
import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from step4_langchain import llm_with_tools   # 🔴 Step 4-ലെ LLM + tools
from step5_langgraph import tools_node       # 🔴 Step 5-ലെ നമ്മുടെ സ്വന്തം tools node

# ================= STRICTER PROMPT (hallucination fix) =================
SYSTEM_PROMPT = """You are CineSense, a friendly movie assistant.
Your ONLY data source is the TMDB 5000 dataset (mostly Hollywood movies up to 2017), which you access through your tools.
Rules:
- Only mention movies that appear in tool results in this conversation.
- Use ratings, years and other numbers exactly as the tools return them. Never use IMDb ratings or your own memory.
- If the data doesn't have what the user wants, say so honestly.
- Keep answers short and friendly."""   # 🔴 ചേട്ടന്റെ കർശന നിയമം: tool results മാത്രം, സ്വന്തം ഓർമ്മ വേണ്ട


# ================= BUILD AGENT (once) =================
@st.cache_resource   # 🔴 agent-ഉം memory-യും ഒരിക്കൽ മാത്രം ഉണ്ടാക്കുന്നു, ഓരോ click-ലും വീണ്ടും ഉണ്ടാക്കില്ല
def build_agent():
    def agent_node(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())   # 🔴 Step 5-ലെ graph + Step 6-ലെ memory


agent = build_agent()


def ask_agent(question):
    config = {"configurable": {"thread_id": st.session_state.thread_id}, "recursion_limit": 12}
    tools_used, answer = [], ""
    try:
        for update in agent.stream({"messages": [HumanMessage(content=question)]}, config, stream_mode="updates"):
            for node_name, data in update.items():
                if node_name == "agent":
                    msg = data["messages"][-1]
                    if msg.tool_calls:
                        tools_used += [f"{tc['name']}({tc['args']})" for tc in msg.tool_calls]   # 🔴 ഏതൊക്കെ tools ഉപയോഗിച്ചു എന്ന് ശേഖരിക്കുന്നു, UI-ൽ കാണിക്കാൻ
                    else:
                        answer = msg.content
    except GraphRecursionError:
        answer = "Sorry, that took too many steps. Please try a simpler question."
    except Exception as e:
        answer = f"Sorry, something went wrong ({type(e).__name__}). Please try again."
    return answer, tools_used


def show_tools(tools_used):
    if tools_used:
        with st.expander(f"🔧 Tools used ({len(tools_used)})"):   # 🔴 agent ഉള്ളിൽ ചെയ്തത് കാണിക്കുന്ന ചെറിയ പെട്ടി: demo-യ്ക്ക് വളരെ ഉപകാരപ്രദം
            for t in tools_used:
                st.code(t, language=None)


# ================= PAGE =================
st.set_page_config(page_title="CineSense Agent", page_icon="🎬")
st.title("🎬 CineSense Agent")
st.caption("An AI movie assistant that uses my own ML models (recommender + rating predictor) as tools · LangGraph + Groq")

if "thread_id" not in st.session_state:                 # 🔴 ഓരോ browser tab-നും സ്വന്തം conversation (thread_id)
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:                   # 🔴 screen-ൽ കാണിക്കാനുള്ള chat history
    st.session_state.history = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Try asking")
    examples = [
        "Recommend 3 movies like The Dark Knight",
        "Movies like Interstellar rated above 7.5",
        "Best science fiction movies",
        "Predict the rating of a 2 hour thriller with a 50 million dollar budget",
    ]
    clicked = None
    for q in examples:
        if st.button(q, use_container_width=True):
            clicked = q
    st.divider()
    if st.button("🗑️ New chat", use_container_width=True):   # 🔴 പുതിയ thread_id = പുതിയ conversation, memory ഇല്ല
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.caption("Data: TMDB 5000 (movies up to 2017)")

# ---------- Old messages ----------
for item in st.session_state.history:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        show_tools(item.get("tools"))

# ---------- New message ----------
prompt = st.chat_input("Ask me about movies...") or clicked   # 🔴 type ചെയ്തതോ sidebar-ൽ click ചെയ്തതോ ആയ ചോദ്യം

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🍿"):
            answer, tools_used = ask_agent(prompt)
        st.markdown(answer)
        show_tools(tools_used)

    st.session_state.history.append({"role": "assistant", "content": answer, "tools": tools_used})