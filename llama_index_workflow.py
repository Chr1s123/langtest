import os
import asyncio
import typing
from dotenv import load_dotenv

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step
)
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

# 数据事件定义
class DraftReqEvent(Event):
    category: str
    inquiry: str
    feedback: str = ""

class DraftResultEvent(Event):
    inquiry: str
    category: str
    draft: str


class TicketWorkflow(Workflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = OpenAILike(
            model=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"),
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            api_base=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        )

    @step
    async def classify(self, ev: StartEvent) -> DraftReqEvent:
        inquiry = ev.inquiry
        print(f"\n[1] 开始分类工单: {inquiry}")
        prompt = f"请将以下用户工单分类到 ['billing', 'tech', 'other'] 之一。工单内容: {inquiry}。只需要回复类别名称，不要输出其他任何内容。"
        response = await self.llm.acomplete(prompt)
        category = str(response).strip().lower()
        if category not in ['billing', 'tech', 'other']:
            category = 'other'
        print(f"    --> 分类结果: {category}")
        return DraftReqEvent(category=category, inquiry=inquiry, feedback="")

    @step
    async def draft_response(self, ev: DraftReqEvent) -> DraftResultEvent:
        print(f"[2] 起草回复 (类别={ev.category})...")
        if ev.feedback:
            print(f"    收到之前的修改建议: {ev.feedback}")
            
        sys_prompt = "你是一个客服。"
        if ev.category == 'billing':
            sys_prompt += "请解释账单并提供核对指引。"
        elif ev.category == 'tech':
            sys_prompt += "请提供技术排查步骤。"
            
        prompt = f"{sys_prompt}\n\n工单:{ev.inquiry}\n"
        if ev.feedback:
            prompt += f"\n注意，之前的草稿未能通过审核，修改意见为: {ev.feedback}\n请根据意见重新拟定并改进语气。"
            
        response = await self.llm.acomplete(prompt)
        draft = str(response)
        print(f"    --> 起草完成 (截取): {draft[:30]}...")
        return DraftResultEvent(inquiry=ev.inquiry, category=ev.category, draft=draft)

    @step
    async def review_response(self, ev: DraftResultEvent) -> typing.Union[DraftReqEvent, StopEvent]:
        # ...
        print(f"[3] 审核回复内容...")
        prompt = f"请严格审核以下客服回复草稿。要求：【必须极其友善，必须包含敬语】。\n草稿：'{ev.draft}'。\n如果极其友善且合格，请回复完全精确匹配的 'PASS'；如果不合格或者不够友善，请提供一句话的修改建议。"
        response = await self.llm.acomplete(prompt)
        result = str(response).strip()
        
        if result == 'PASS' or result.startswith('PASS'):
            print("    --> 审核通过！")
            return StopEvent(result=ev.draft)
        else:
            print(f"    --> 审核未通过，发回重写。建议: {result}")
            # 发送回退事件，形成 Workflow 循环
            return DraftReqEvent(category=ev.category, inquiry=ev.inquiry, feedback=result)

async def main():
    w = TicketWorkflow(timeout=60, verbose=False)
    inquiry = "我上个月的信用卡账单乱七八糟的，是不是你们系统出错了？快给我查查！"
    result = await w.run(inquiry=inquiry)
    print("\n================ [最终输出] ================\n")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
