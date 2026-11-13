from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .config import LLM_MODEL, TEMPERATURE, check_config
from .prompt import (
    HR_AGENT_PROMPT,
    OUT_OF_SCOPE_MESSAGE,
    POLICY_ANSWER_PROMPT,
    REJECTION_CONTEXT_SUFFIX,
    ROUTER_PROMPT,
)
from .rag import build_retriever, format_docs
from .state import HRState, Route
from .tools import HR_TOOLS

check_config()

llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)

retriever = build_retriever()


def latest_question(state: HRState) -> str:
    """The most recent thing the employee typed."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


#Node 1
class RouteDecision(BaseModel):
    """The router's decision. with_structured_output forces the LLM to fill this."""

    route: Route = Field(description="The single route this question belongs to.")
    reason: str = Field(description="One short sentence explaining the choice.")


router_llm = llm.with_structured_output(RouteDecision)


def router(state: HRState) -> dict:
    decision = router_llm.invoke(
        [SystemMessage(content=ROUTER_PROMPT), HumanMessage(content=latest_question(state))]
    )
    # Reset the per-question keys so every new question starts clean.
    return {
        "route": decision.route,
        "policy_context": "",
        "rejection_explained": False,
    }



# NODE 2

def retrieve_policy(state: HRState) -> dict:
    question = latest_question(state)
    update = {}

    if state.get("route") == "hr_action":
        # We got here because a validation tool said valid=False.
        # Search with the rejection reason so we fetch the RULE behind it.
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage):
                question = f"{question}\n{message.text}"
                break
        update["rejection_explained"] = True

    documents = retriever.invoke(question)
    update["policy_context"] = (
        format_docs(documents) or "No relevant section was found in the Employee Handbook."
    )
    return update



# NODE 3
def policy_answer(state: HRState) -> dict:
    system = SystemMessage(
        content=POLICY_ANSWER_PROMPT.format(context=state.get("policy_context", ""))
    )
    response = llm.invoke([system, *state["messages"]])
    return {"messages": [response]}


# NODE 4
llm_with_tools = llm.bind_tools(HR_TOOLS)

llm_answer_only = llm.bind_tools(HR_TOOLS, tool_choice="none")


def hr_agent(state: HRState) -> dict:
    if state.get("rejection_explained"):
        prompt = HR_AGENT_PROMPT + REJECTION_CONTEXT_SUFFIX.format(
            context=state.get("policy_context", "")
        )
        response = llm_answer_only.invoke([SystemMessage(content=prompt), *state["messages"]])
    else:
        response = llm_with_tools.invoke(
            [SystemMessage(content=HR_AGENT_PROMPT), *state["messages"]]
        )
    return {"messages": [response]}

# Node 5
tools = ToolNode(HR_TOOLS)


# NODE 6

def out_of_scope(state: HRState) -> dict:
    return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}
