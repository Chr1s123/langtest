# 三大编排架构核心代码解析

本文档针对项目中的三个执行链路脚本进行逐块、跨行级别的解析。你将清晰地看到这三种框架是如何组织代码、怎样传递状态参数的，以及面对“**自我纠错循环重试**”这种智能体最核心的行为时，三者的代码有何不同的应对姿态。

---

## 一、LangChain LCEL 声明式链路 (`langchain_lcel.py`)

LangChain 的 Expression Language (LCEL) 提倡极其精简优雅的“单向流通链”设计，它的核心是通过类 Linux 管道 `|` 来拼接所有的业务动作。

### 1. 模型初始化
```python
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

load_dotenv()

# 初始化兼容 OpenAI 的 LLM 对象。
# 这里会自动挂载我们在 .env 配置好的基础 URL 和模型字符串，成为后续所有节点的调用引擎。
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
)
```

### 2. 定义分类和起草节点
对于 LCEL，每一个处理步骤最终都会被转化为一个 `Runnable` 对象。输入通常是字典类型的 `state`。
```python
def classify_inquiry(state):
    inquiry = state["inquiry"]
    prompt = f"请将以下用户工单分类到 ['billing', 'tech', 'other'] 之一。工单内容: {inquiry}。只需要回复类别名称，不要输出其他内容。"
    
    # 阻塞式调用模型。用 .content 取出文本答复。
    category = llm.invoke(prompt).content.strip().lower()
    if category not in ['billing', 'tech', 'other']:
        category = 'other'
        
    # Python **state 将上游的数据原封不动继承，同时覆盖添加本次分析出的 category（类别）返回
    return {**state, "category": category}

def draft_response(state):
    # 根据上一步分类好的 category 和之前可能被驳回的 feedback，组织最终 Prompt
    category = state["category"]
    inquiry = state["inquiry"]
    feedback = state.get("feedback", "")
    
    sys_prompt = "你是一个客服。"
    # ... 省略部分 prompt 拼接代码 ...
        
    draft = llm.invoke(prompt).content
    # 返回最新写好的 draft
    return {**state, "draft": draft}
```

### 3. 应对“循环打回”时的痛点
LCEL 的链是 `a | b | c` 往前走的抽象，并没有“让流反向重走某几截”的内置接口。这导致我们无法用 LCEL 原生的风格处理循环，**通常只能写一个包含了普通的 Python `while` 循环节点的突兀结构。**
```python
def review_loop(state):
    max_retries = 3
    retries = 0
    current_state = state
    
    while retries < max_retries:
        # 手动去调用上面定义的起草节点函数
        current_state = draft_response(current_state)
        draft = current_state["draft"]
        
        prompt = f"请严格审核以下客服回复草稿 [...]"
        review_result = llm.invoke(prompt).content.strip()
        
        if review_result == 'PASS' or review_result.startswith('PASS'):
            # 退出循环，成功拦截
            return current_state
        else:
            # 修改输入参数，使得下一轮 while 循环的 draft_response() 能捕捉到 feedback
            current_state["feedback"] = review_result
            retries += 1
            
    return current_state
```

### 4. 链路组装
```python
# 将普通的 Python 函数包通过 RunnableLambda 包装后，利用 | 拼接
# 管道化看起来十分干净
chain = (
    RunnableLambda(classify_inquiry)
    | RunnableLambda(review_loop)
)

if __name__ == "__main__":
    result = chain.invoke({"inquiry": "测试..."})
```

---

## 二、LangGraph 显式图状态机 (`langgraph_app.py`)

LangGraph 可以被理解为用来解决 LCEL “无法建立回退跳转机制和复杂死循环” 的增强架构，它完全采用“图论”的思想。

### 1. 结构化类状态与节点定义
与 LCEL 纯使用字典不同，它要求你显式声明图级别的全局状态：
```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# TypedDict 保证状态参数在任意一个图中扭转时都不会出现 Key 未知丢失的情况
class GraphState(TypedDict):
    inquiry: str
    category: str
    draft: str
    feedback: str
    retry_count: int

def node_classify(state: GraphState):
    # 此处逻辑与上方一致，略... 
    # 但是我们不需要 {**state, ...} 承袭旧状态数据，LangGraph 会在底层框架处理字段的 Update 自动合并。
    return {"category": category, "retry_count": 0, "feedback": ""}

def node_draft(state: GraphState):
    # 略...
    return {"draft": draft}
    
def node_review(state: GraphState):
    # 略... 
    # 每次打回时，将原有的 retry_count 数字加 1
    return {"feedback": review_result, "retry_count": retry_count + 1}
```

### 2. 路由枢纽：完美的逻辑控制
所有的流转不仅是按照代码编写的行数定死的，而是根据专门的路由逻辑实现重试功能：
```python
# 我们不写死 while 循环了，而是把要不要重来一次交给下面这个路由纯函数。
# 它根据状态判断输出应该跳去哪
def review_router(state: GraphState):
    if state["feedback"] == "PASS":
        return "end"
    elif state.get("retry_count", 0) >= 3:
        return "end"
    else:
        return "retry"
```

### 3. 构建和图执行连线（Edges）
通过将业务解构为 Nodes（点）和 Edges（连线），你的应用天然就可以直接被绘制为监控流程图。
```python
# 构建图对象
workflow = StateGraph(GraphState)

# 注册所有点集
workflow.add_node("classify", node_classify)
workflow.add_node("draft", node_draft)
workflow.add_node("review", node_review)

# 常规按顺序跑
workflow.add_edge(START, "classify")
workflow.add_edge("classify", "draft")
workflow.add_edge("draft", "review")

# 这是框架核心的条件边。
# 在 review 节点执行完后，启动 review_router 进行判断：
# 如果路由给出了 retry 值，那么将边强制引流回 "draft" 起草节点；如果给出 end 则前往结束(END)。
workflow.add_conditional_edges(
    "review",
    review_router,
    {
        "retry": "draft",
        "end": END
    }
)

app = workflow.compile()
```

---

## 三、LlamaIndex Workflows 事件驱动总线 (`llama_index_workflow.py`)

LlamaIndex 原本是个 RAG 做检索的库，但它在最新的版本主推了通过“Event（事件）” 抛掷实现的纯解耦智能体设计。

### 1. 结构化事件声明 (Event)
在这个框架下，你不再有全局共享的一致 `State` 了。前一个动作必须向后广播“发生了什么事且携带什么数据”。
```python
from llama_index.core.workflow import Event, StartEvent, StopEvent, Workflow, step

# 我们定义了两个自发事件数据容器
class DraftReqEvent(Event):
    # 想要让别人起草回复，就向他发出这个带有品类和要求的数据结构
    category: str
    inquiry: str
    feedback: str = ""

class DraftResultEvent(Event):
    # 起草完毕后，广播一个带有这三个参数的结束事件
    inquiry: str
    category: str
    draft: str
```

### 2. 发布与订阅（@step 钩子）
每一个具有 `@step` 装饰器的类函数都在无底线地监听和它**签名形参**一致的请求事件。
```python
class TicketWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 支持各种 OpenAI 兼容底层模型的快速接入接口包装
        self.llm = OpenAILike( ... )

    # classify 函数在静默监听 StartEvent 发出的情况。
    # 一旦外部调用系统了，它会自动捕获，在计算完毕后，return 出自己扔向天空的 DraftReqEvent。
    @step
    async def classify(self, ev: StartEvent) -> DraftReqEvent:
        # 此时发生了 prompt 并发了 LLM，逻辑略...
        
        # 自己计算完以后啥也不用管，发一个广播(返回)就完了。
        return DraftReqEvent(category=category, inquiry=inquiry, feedback="")

    # 这是第二个干活的点，它并不在乎是谁调用的它。
    # 只要系统内出现 DraftReqEvent ，无论是分类(Classify)抛出来的，还是后面被打回重写抛出来的，它都会全数照单全收并处理。
    @step
    async def draft_response(self, ev: DraftReqEvent) -> DraftResultEvent:
        # LLM 计算，代码略... 
        
        # 完事了再向天空中抛出一个自己处理完了的 DraftResultEvent 事件
        return DraftResultEvent(inquiry=ev.inquiry, category=ev.category, draft=draft)
```

### 3. 解除时间轴，实现重试事件抛掷
在 LlamaIndex Workflow 完全是并行的事件引擎。审核打回逻辑简直是最天然的适配场景：
```python
    # 监听上一步的产物事件
    @step
    async def review_response(self, ev: DraftResultEvent) -> typing.Union[DraftReqEvent, StopEvent]:
        # 调用大模型验证合法性，代码略... 
        
        if result == 'PASS':
            # 如果审核通过，直接广播 StopEvent 强行停止这个大工作流。
            return StopEvent(result=ev.draft)
        else:
            # 【重要转场】：如果不通过怎么办？
            # 极其精髓：我们再次利用 return 扔一个要求写回复的 DraftReqEvent 出去即可！
            # 那么系统内所有时刻监听 DraftReqEvent 的节点（也就是我们上方的 draft_response 函数）
            # 就又会再次收到召唤并开始二次重干活，实现了彻底解耦的高等重试闭环。
            return DraftReqEvent(category=ev.category, inquiry=ev.inquiry, feedback=result)

# LlamaIndex 要求显式配置异步 asyncio 系统驱动：
async def main():
    w = TicketWorkflow(timeout=60, verbose=False)
    # 起源触发
    result = await w.run(inquiry="测试输入") 
```
