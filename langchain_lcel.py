import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
)

def classify_inquiry(state):
    inquiry = state["inquiry"]
    print(f"\n[1] 开始分类工单: {inquiry}")
    prompt = f"请将以下用户工单分类到 ['billing', 'tech', 'other'] 之一。工单内容: {inquiry}。只需要回复类别名称，不要输出其他内容。"
    
    # 直接调用大模型
    category = llm.invoke(prompt).content.strip().lower()
    if category not in ['billing', 'tech', 'other']:
        category = 'other'
        
    print(f"    --> 分类结果: {category}")
    return {**state, "category": category}

def draft_response(state):
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
    return {**state, "draft": draft}

def review_loop(state):
    """
    注意：这是 LCEL 架构的一大痛点。
    对于需要“自我审视如果不通过则返回上游”的循环控制，LCEL 缺乏原生的语义支持。
    通常只能退化成将节点包裹在一个常规的 Python while 循环中，打破了声明式的特性。
    """
    max_retries = 3
    retries = 0
    current_state = state
    
    while retries < max_retries:
        # 手动调用上游重试逻辑
        current_state = draft_response(current_state)
        draft = current_state["draft"]
        
        print(f"[3] 审核回复内容...")
        prompt = f"请严格审核以下客服回复草稿。要求：【必须极其友善，必须包含敬语】。\n草稿：'{draft}'。\n如果极其友善且合格，请回复完全精确匹配的 'PASS'；如果不合格或者不够友善，请提供一句话的修改建议。"
        review_result = llm.invoke(prompt).content.strip()
        
        if review_result == 'PASS' or review_result.startswith('PASS'):
            print("    --> 审核通过！")
            return current_state
        else:
            print(f"    --> 审核未通过，发回重写。建议: {review_result}")
            current_state["feedback"] = review_result
            retries += 1
            
    print("    --> 达到最大重试次数，强制输出最后一次草稿。")
    return current_state

# 创建 LCEL 纯净管道
chain = (
    RunnableLambda(classify_inquiry)
    | RunnableLambda(review_loop)
)

if __name__ == "__main__":
    inquiry = "我上个月的信用卡账单乱七八糟的，是不是你们系统出错了？快给我查查！"
    result = chain.invoke({"inquiry": inquiry, "feedback": ""})
    print("\n================ [最终输出] ================\n")
    print(result.get("draft", "无"))
