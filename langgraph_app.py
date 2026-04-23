import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
)

# 1. 定义强类型的图状态
class GraphState(TypedDict):
    inquiry: str
    category: str
    draft: str
    feedback: str
    retry_count: int

# 2. 定义各个图节点(Nodes)
def node_classify(state: GraphState):
    inquiry = state["inquiry"]
    print(f"\n[1] 开始分类工单: {inquiry}")
    prompt = f"请将以下用户工单分类到 ['billing', 'tech', 'other'] 之一。工单内容: {inquiry}。只需要回复类别名称，不要输出其他任何内容。"
    category = llm.invoke(prompt).content.strip().lower()
    if category not in ['billing', 'tech', 'other']:
        category = 'other'
    print(f"    --> 分类结果: {category}")
    
    # 状态合并返回
    return {"category": category, "retry_count": 0, "feedback": ""}

def node_draft(state: GraphState):
    category = state["category"]
    inquiry = state["inquiry"]
    feedback = state.get("feedback", "")
    print(f"[2] 起草回复 (类别={category})...")
    
    sys_prompt = "你是一个客服。"
    if category == 'billing':
        sys_prompt += "请解释账单并提供核对指引。"
    elif category == 'tech':
        sys_prompt += "请提供技术排查步骤。"
        
    prompt = f"{sys_prompt}\n\n工单:{inquiry}\n"
    if feedback:
        print(f"    收到之前的修改建议: {feedback}")
        prompt += f"\n注意，之前的草稿未能通过审核，修改意见为: {feedback}\n请根据意见重新拟定并改进语气。"
        
    draft = llm.invoke(prompt).content
    print(f"    --> 起草完成 (截取): {draft[:30]}...")
    return {"draft": draft}

def node_review(state: GraphState):
    draft = state["draft"]
    retry_count = state.get("retry_count", 0)
    print(f"[3] 审核回复内容 (当前重试次数: {retry_count})...")
    
    prompt = f"请严格审核以下客服回复草稿。要求：【必须极其友善，必须包含敬语】。\n草稿：'{draft}'。\n如果极其友善且合格，请回复完全精确匹配的 'PASS'；如果不合格或者不够友善，请提供一句话的修改建议。"
    review_result = llm.invoke(prompt).content.strip()
    
    if review_result == 'PASS' or review_result.startswith('PASS'):
        print("    --> 审核通过！")
        return {"feedback": "PASS"}
    else:
        print(f"    --> 审核未通过，发回重写。建议: {review_result}")
        return {"feedback": review_result, "retry_count": retry_count + 1}

# 3. 编排边流转逻辑
def review_router(state: GraphState):
    """
    此路由函数优雅地解决了循环验证难题
    """
    if state["feedback"] == "PASS":
        return "end"
    elif state.get("retry_count", 0) >= 3:
        print("    --> 达到最大重试次数，跳出循环。")
        return "end"
    else:
        return "retry"

# 4. 构建并编译 StateGraph
workflow = StateGraph(GraphState)

workflow.add_node("classify", node_classify)
workflow.add_node("draft", node_draft)
workflow.add_node("review", node_review)

# 常规按顺序流转
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "draft")
workflow.add_edge("draft", "review")

# LangGraph 最核心特性：添加条件边控制连线和死循环回退
workflow.add_conditional_edges(
    "review",
    review_router,
    {
        "retry": "draft", # 回退重写
        "end": END        # 正确流转结束
    }
)

app = workflow.compile()

if __name__ == "__main__":
    inquiry = "我上个月的信用卡账单乱七八糟的，是不是你们系统出错了？快给我查查！"
    final_state = app.invoke({"inquiry": inquiry, "retry_count": 0})
    print("\n================ [最终输出] ================\n")
    print(final_state.get("draft", "无"))
