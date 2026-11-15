from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from .edges import after_hr_tools, after_retrieval, route_question
from .nodes import hr_agent, out_of_scope, policy_answer, retrieve_policy, router, tools
from .state import HRState

graph = StateGraph(HRState)

# ---- 1. Nodes -------------------------------------------------------------
graph.add_node("router", router)
graph.add_node("retrieve_policy", retrieve_policy)
graph.add_node("policy_answer", policy_answer)
graph.add_node("hr_agent", hr_agent)
graph.add_node("tools", tools)
graph.add_node("out_of_scope", out_of_scope)


graph.add_edge(START, "router")
graph.add_conditional_edges("router",route_question,{"policy_question": "retrieve_policy",
                                                     "hr_action": "hr_agent",
                                                     "out_of_scope": "out_of_scope"})

graph.add_conditional_edges("retrieve_policy",after_retrieval, {"policy_answer": "policy_answer",
                                                                "hr_agent": "hr_agent"})

graph.add_conditional_edges("hr_agent", tools_condition, {"tools": "tools", END: END})

graph.add_conditional_edges("tools",after_hr_tools,{"hr_agent": "hr_agent",
                                                    "explain_rejection": "retrieve_policy"})

graph.add_edge("policy_answer", END)
graph.add_edge("out_of_scope", END)

app = graph.compile()
