# 第2章. LangChain入门

本章我们会分成2个部分：

1. LangChain核心组件

1. Agent实战案例

---

## 第1节. LangChain核心组件

### 1. 认识**LangChain**

LangChain 由 Harrison Chase 创建于2022年10月，是用于开发**智能体工程（Agent Engineering）**的平台**。**

官网地址：

[LangChain](https://www.langchain.com/)

官网文档：

[Home - Docs by LangChain](https://docs.langchain.com/)

#### 1.1 架构体系

**LangChain**并不仅仅是一个框架，而是一整个智能体开发平台，包含很多不同的组件。

其中，包含一系列开源的**智能体（Agent）**开发框架，而且兼容Python和TypeScript两种语言：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/WfcNbPjeBoCOKjxhwmkceEAUn5e/)

- LangChain：用于快速构建智能体，可兼容任何模型提供商。

- LangGraph：从底层一步步控制智能体的构建，包括记忆（Memory）、人机协同（HITL）等

- Deep Agents：用于构建复杂的、处理多步骤的任务的智能体

另外，LangChain还包含一套帮助人工智能团队利用实时生产数据进行**持续测试和改进**的平台，叫做LangSmith：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BB0TbltQzoR11yxPjdBc62Hqnke/)

 
> 总结：
> LangChain是智能体开发平台，包含一套各种帮助开发、测试、评估智能体的框架。核心包括：
>   - LangChain：用于快速构建智能体，可兼容任何模型提供商。
>   - LangGraph：从底层一步步控制智能体的构建，包括记忆（Memory）、人机协同（HITL）等

可以看到，LangChain平台的所有框架都是围绕着构建智能体（Agent）这一目标的，那么问题来了：

> **什么是智能体（Agent）呢？**

#### 1.2 什么是Agent

什么是Agent，这其实没有一个标准的答案，每个人都有自己的理解。

对于这个问题，LangChain创始人[Harrison Chase](https://www.blog.langchain.com/tag/harrison-chase/)有一个偏向技术性的答案：

> An AI agent is a system that uses an LLM to decide the control flow of an application.
> Agent是一种使用大语言模型（LLM）来决定应用程序控制流的系统。

在人工智能领域，**Agent**（通常翻译为**智能体**或**代理**）是指**一种能够感知环境、进行推理、自主决策并采取行动以实现特定目标的智能系统。**

| 特性 | 传统聊天机器人/LLM | AI Agent |
| --- | --- | --- |
| 交互模式 | 被动响应，问一句答一句 | 主动规划，以目标为导向 |
|  |  |  |

如果说大模型（LLM）是“大脑”，那么 Agent 就是**“拥有手脚和思维逻辑的独立个体”**。它不再只是被动地回答问题，而是能主动拆解任务并调用各种工具来完成工作。

例如：要开发一个《AI旅游助手》的应用。

如果是**传统LLM应用**，程序流程是这样的：

> 1. 用户提出需求，例如：帮我计划一个5天的北京之旅，预算8000元，我喜欢历史。
> 1. 调用LLM，分析用户需求，直接由LLM生成一个简单旅游计划

这个计划基于它训练数据中的通用知识，可能没有考虑当前的天气、景点是否关闭、门票是否可预订等实时信息。

如果是**Agent应用**，Agent可以自主规划程序流程：

> 1. 用户提出需求，例如：帮我计划一个5天的北京之旅，预算8000元，我喜欢历史。
> 1. Agent分析用户需求，分步执行：

Agent通过主动规划任务流程，主动使用工具，整合了实时信息，并进行了动态调整，最终产出的是一个真正可落地的方案。

总结如下：

- LLM = 聪明的大脑

- Agent = 聪明的大脑 + 手脚

当然，Agent的模式也是在不断演进的：

- 阶段一：ReAct + Tool Calling

- 阶段二：Reflection + Long Memory

- 阶段三：Multi Agent System，MAS

接下来，我们会从最简单的Agent开始学习，逐渐升级到更复杂的Agent结构。

#### 1.3 快速入门

下面，我们通过一个快速入门，了解Agent的定义和工作流程。

##### 1.3.1 准备工作

首先，要使用LangChain必须先安装依赖，命令如下：

```Python
uv add langchain
```

LangChain支持各种不同的模型，而且提供了对应的兼容SDK，不过也都需要安装对应依赖，你可以按需添加：

```Python
# 集成 DeepSeek
uv add langchain-deepseek
```

##### 1.3.2 代码示例

接下来就可以开发Agent了，基本步骤如下：

1. 加载环境变量

1. 定义工具

1. 定义Agent

1. 调用Agent

Langchain提供了**create_agent**方法用来快速创建Agent，我们只需要提供好Agent所需的**模型（Models）**、**工具（Tools）**即可。

示例代码如下：

```Python
# 1.加载环境变量
```

运行结果如下：

```Python
正在调用大模型...
{'messages': [HumanMessage(content='杭州今天天气如何?', additional_kwargs={}, response_metadata={}, id='216c9cd1-8ebc-4365-a192-6b1a30ae788c'), AIMessage(content='我来帮您查询杭州今天的天气情况。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 51, 'prompt_tokens': 313, 'total_tokens': 364, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 57}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': 'bbeda11e-7653-4c3d-9cc5-9a58491f63f0', 'finish_reason': 'tool_calls', 'logprobs': None}, id='lc_run--019c92f4-8395-7852-8e36-d4645f86d443-0', tool_calls=[{'name': 'getWeather', 'args': {'location': '杭州'}, 'id': 'call_00_H7Yklbf4osnSeFOj3k4TP33N', 'type': 'tool_call'}], invalid_tool_calls=[], usage_metadata={'input_tokens': 313, 'output_tokens': 51, 'total_tokens': 364, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}}), ToolMessage(content='Current weather in 杭州 is sunny', name='getWeather', id='911eda0e-a5a8-4375-909d-b8707b3a08a9', tool_call_id='call_00_H7Yklbf4osnSeFOj3k4TP33N'), AIMessage(content='根据查询结果，杭州今天的天气是**晴朗**的。', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 13, 'prompt_tokens': 388, 'total_tokens': 401, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 68}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '692c1420-4a06-4080-adf7-54250207e86a', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019c92f4-8d74-7770-89fd-da1e9d67efca-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 388, 'output_tokens': 13, 'total_tokens': 401, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}})]}
```

原本大模型不具备查询天气的能力，所以无法回答天气问题。但是，当我们提供了一个查询添加的Tool以后，它就能自动查询天气来回答问题，是不是很神奇。

那么，Agent是如何做到的呢？

传统的LLM应用都是一问一答的形式，模型只能根据自己的训练数据来回答，流程非常简单：

> [白板/画板内容] (token: VMkGdRQHXoS9WgxdMX0cMn5xniY)

而智能体则可以调用工具与外界交互，获取实时信息，工作流程则要复杂很多，是这样的：

> [白板/画板内容] (token: AXGSdOCaUo8y5MxjczvckAn1nHc)

流程如下：

1. 用户提问（Input）：杭州今天天气如何？

1. 模型分析（Reasoning）：用户询问杭州天气，我不知道，需要调用查询天气的工具`get_weather`

1. 调用工具（Action）：调用工具，get_weather，传入城市"杭州"

1. 分析结果（Observation）：工具返回结果，模型分析结果，判断是否足以回答用户问题

1. 生成结果（Output）：根据工具的结果生成响应给用户

那么，模型是如何知道工具的信息的呢？

其实，在大模型提供的API接口中，有一个tools参数，描述了工具的详细信息：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/HsqDbJbrGoUIHjx6JY8chq5Qn8d/)

所以，LangChain会帮助我们把tool的信息封装为此tool参数，与message一起发送给大模型，大模型就了解tool的详细信息，根据用户需求判断是否需要调用tool，需要调用哪个tool.

那么问题来了，当大模型决定调用某个tool时，该如何调用呢？毕竟，tool是我们定义的，模型是没有调用能力的。

模型确实不能直接调用tool，只能返回字符串。但是它可以把要调用的tool信息、参数信息都以Json格式返回：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BR3hb9UNioYipoxBXH3c4oVNnac/)

这样一来，LangChain就会帮我们解析响应结果中的Function信息，也就是tool信息，就知道了要调用哪个函数，以及参数是什么了。LangChain就会执行该函数，再把得到的结果再次发送给大模型。

具体的工作流程如图：

> [白板/画板内容] (token: VqQWdDkUToq6g5xW4kocRSQpnXd)

OK，弄明白了Agent的原理，我们不难发现，Agent中最重要的两个部分，就是：

- Model：负责推理分析、思考，相当于Agent的大脑

- Tools：负责执行任务，相当于Agent与外界交互的手脚

当然，Agent中肯定不止这两个部分，接下来，我们就逐一解析Agent创建的各个细节。

### 2. 模型（Models）

这里说的模型，完整叫法是大语言模型（LLM）。它能够理解人类语言，使用人类语言生成内容、翻译、提取摘要、回答问题等。

不仅如此，现在大多数的模型还有一些特别能力：

- [Tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling) - 调用外部工具（例如查询数据库或调用 API），并在其回复中使用这些工具返回的结果。

- [Structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output) - 将模型的响应结果约束为遵循已定义的格式，例如：json

- [Multimodality](https://docs.langchain.com/oss/python/langchain/models#multimodal) - 可以处理和返回文本以外的数据，如图像、音频和视频。

- [Reasoning](https://docs.langchain.com/oss/python/langchain/models#reasoning) - 模型可以执行多步推理来得出结论。

可以说LLM就是Agent的大脑，是Agent的推理引擎。它驱动Agent做出每个决定：何时调用工具、调用哪个工具、如何解释结果，以及何时提供最终答案。

LangChain支持现在市面上大部分的大语言模型（LLM），并且提供了统一的模型调用接口。使您可以轻松访问许多不同的模型提供者，并且在模型之间进行试验和切换也变得很容易。

 
> 有关模型提供者（Model Providers）的信息和功能，请参阅langchain官网的 [chat model page](https://docs.langchain.com/oss/python/integrations/chat).

#### 2.1 初始化模型

langchain提供了两种常见方法用来初始化模型：

- 使用`init_chat_model`函数，由langchain自动创建模型对象

- 使用不同模型对应的Model类，手动创建模型对象

##### 2.1.1 init_chat_model

在LangChain中开始使用独立模型的最简单方法是使用`init_chat_model`函数。

调用`init_chat_model`函数时，你需要从langchain支持的**模型提供者**（**Model Provider**）中选择一个模型，而langchain会自动初始化这个模型，非常方便。

例如，我们要使用Deepseek这个模型。

- **首先**，我们需要安装模型依赖：

- **然后**，我们要确保在项目的**.env**环境中配置好**api_key**:

- **最后**，就可以直接使用init_chat_model初始化模型了：

- **测试**，我们可以通过打印model的类型，查看生成的结果：

可见，采用`init_chat_model`自动初始化模型时，模型的类型由LangChain通过模型名称自动推断。

如果要切换其它模型，我们只需要安装其它模型依赖，然后配置API_KEY，改变模型名称即可，其它代码不用动。

##### 2.1.2 自定义模型及参数

`init_chat_model`默认会根据模型名称自动确定**模型的提供者**、其`base_url`，并从env读取`api_key`，但前提是必须是langchain支持的模型提供者([支持模型参考链接](https://reference.langchain.com/python/langchain/chat_models/base/init_chat_model))，例如：

- Openai

- Deepseek

- Google

- Anthropic

- ...

对于其它不支持的模型，我们必须自定义模型参数来访问。

例如，我们要访问阿里云百炼的qwen-max，它就是不被langchain支持的模型，我们必须自定义模型参数来访问。

- 我们需要在环境变量中定义**api_key**和**base_url**

- 然后在`init_chat_model`中指定**model**、**model_provider**、**base_url**和**api_key**

具体步骤如下：

- **首先，**在.env中配置好`api_key`和`base_url`：

- **然后，**手动读取环境变量中的`api_key`和`base_url`：

- **最后，**调用init_chat_model，初始化模型：

- **测试**，查看生成的模型类型：

可见，通过参数自定义模型时，模型的类型由`model_provider`参数类决定。

除了修改模型提供者以外，`init_chat_model`方法允许我们调整模型参数，例如：

- temperature: 控制生成文本的随机性，值越小越确定，值越大越随机

- max_tokens: 控制生成文本的最大长度

- top_p: 控制生成文本的多样性，值越小越多样，值越大越确定

- timeout: 控制生成文本的超时时间

- max_retries: 控制生成文本的最大重试次数

- ...

示例：

```Python
# 调用init_chat_model函数初始化模型，并设定模型参数
model = init_chat_model(
    model="qwen-max", 
    model_provider="openai",
    base_url=base_url,
    api_key=api_key,
    temperature=1.5,
)
```

##### 2.1.3 使用Model类

其实`init_chat_model`方法底层就是帮我们利用Model类创建对象。但只支持有限的模型。而在langchain的社区，除了langchain官方提供的Model，还有些类是社区提供，更丰富多样。

具体支持的模型，可以查看官网地址：[https://docs.langchain.com/oss/python/integrations/chat](https://docs.langchain.com/oss/python/integrations/chat)

例如，我们使用社区版本的Model类来访问阿里云百炼的通义千问模型：


1. 首先，我们需要安装依赖

1. 然后，我们就可以使用Model类初始化模型了

1. 测试，查看生成的模型类型：

#### 2.2 访问模型

LangChain提供了两个不同的方法来访问模型：

- invoke：阻塞式访问

- stream：流式访问

##### 2.2.1 invoke

invoke方法是阻塞式调用，需要等待模型生成全部结果才会返回，等待时间较长。

```Python
# 调用invoke方法
response = model.invoke("月亮的首都是哪里？")

# 查看响应结果
print(response)
```

##### 2.2.2 stream

阻塞式调用需要等待较长时间才能看到AI返回的结果，而流式调用则可以实时看到AI返回的一个个词。

示例：

```Python
# 通过.stream方法实现流式访问
stream = model.stream("月亮的首都是哪里？")

# stream调用返回的结果是一个generator，方便我们循环获取结果
print(type(stream))

# 遍历stream结果，实时打印AI的回复
for chunk in stream:
    print(chunk.content, end="", flush=True)
```

#### 2.3 在Agent中使用模型

Langchain提供了一个`create_agent`方法用来快速创建智能体。当我们创建Agent的时候，可以直接使用创建好的Model，也可以指定模型名，让Langchain自动初始化模型。

##### 2.3.1 创建智能体

1. 创建智能体，指定模型名，由Langchain初始化模型

```Python
from langchain.agents import create_agent

# 1.指定Model名称，由LangChain自动初始化模型
```

1. 创建智能体，并使用创建好的model

```Python
from langchain.agents import create_agent
from langchain_community.chat_models.tongyi import ChatTongyi

# 1.使用Model类初始化模型
model = ChatTongyi(
    model="qwen-plus"
    # 其它模型参数...
)
```

##### 2.3.2 调用智能体

智能体也分为阻塞调用和流式调用两种。

1. 阻塞式调用，使用invoke方法：

1. 流式调用，只需要把调用方式改为`stream`：

要注意，Agent的stream模式同样返回一个generator，但是其结构由`stream_mode`参数决定：

- messages: 返回LLM生成的每一个片段，是一个包含token和metadata的元组（Tuple）

- updates: 返回Agent运行过程中的每一次事件，例如与LLM的对话、工具的调用等

- custom: 返回通过stream writer记录的每一次自定义的输出

如果是为了流式输出AI返回的结果，使用messages模式即可。

#### 2.4 总结

目前为止，我们学习了：

- 如何使用init_chat_model初始化模型

- 如何使用Model类初始化模型

- 如何调用模型

- 如何创建Agent，并在Agent中使用模型

- 如何调用Agent

### 3. 消息（Messages）

在调用模型时，发送给LLM的消息、LLM返回的消息都包含以下几部分内容：

- role：消息所属角色，可以是system、user、assistant

- content：消息的内容

- metadata（可选）：消息的元数据，例如：消息的ID、消耗的token等

之前我们都是自己用dict模拟消息：

```Python
response = agent.invoke({
    "messages": [{"role": "user", "content": "月亮的首都是哪里？"}]
```

这太麻烦了。在LangChain中发送给LLM的消息、LLM返回的消息都统一被封装为BaseMessage，它是中基本的上下文单元。

#### 3.1 消息类型

在LangChain中，我们并不需要自己创建BaseMessage对象，LangChain已经把常见消息根据角色（Role）创建了对应的BaseMessage的子类：

- SystemMessage：role是system，代表系统消息，用于设定模型角色和交互背景

- HumanMessage：role是user，代表用户输入的消息

- AIMessage：role是assistant，代表LLM生成的响应，包含：文本、工具调用、元数据

- ToolMessage：role是tool，代表工具调用时产生的结果

所以，我们可以这样传递消息列表：

```Python
from langchain.messages import HumanMessage, AIMessage
from langchain.agents import create_agent

# 创建Agent
agent = create_agent(model="deepseek-chat")

# 调用Agent，发送消息
response = agent.invoke({
    "messages": [
```

注意看，Agent的返回结果中包含完整的消息列表（Messages）：

```Plain Text
{'messages': [HumanMessage(content='你好，我是虎哥', additional_kwargs={}, response_metadata={}, id='f5703ee9-f567-48d6-8e07-e6ddaf24547e'), AIMessage(content='你好，虎哥，很高兴认识你。', additional_kwargs={}, response_metadata={}, id='5c654447-828c-43b7-9505-a341e0d21b8a', tool_calls=[], invalid_tool_calls=[]), HumanMessage(content='我的名字是什么？', additional_kwargs={}, response_metadata={}, id='a3390334-85b8-4f5f-8528-782a18671ac9'), AIMessage(content='你刚才提到你的名字是“虎哥”。如果这是你希望我称呼你的方式，我会记住的。如果有其他偏好，随时告诉我哦！ ', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 33, 'prompt_tokens': 26, 'total_tokens': 59, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 26}, 'model_provider': 'deepseek', 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '9ea39267-c54a-4523-82e0-1377435ffde4', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019cad79-8b7a-7861-855b-5a87ba11d38c-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 26, 'output_tokens': 33, 'total_tokens': 59, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}
```

我们可以通过遍历Messages数组，更友好的打印结果：

```Python
for message in response['messages']:
    message.pretty_print()
```

结果：

```Plain Text
================================ Human Message =================================

你好，我是虎哥
================================== Ai Message ==================================

你好，虎哥，很高兴认识你。
================================ Human Message =================================
```

 
> **提示**：
> 通过刚才的实现可以发现，拼接message列表可以让AI记住会话历史，产生记忆。不过手动拼接Message太麻烦了，后面我们学习如何实现自动的会话记忆功能。

#### 3.2 多模态消息

之前我们都是向模型发送文本消息，但是 LangChain 也支持向模型发送多模态消息，比如图片、音频、视频、文本等。但前提是必须是多模态模型才支持。

一些支持多模态的模型有：

- qwen3.5-plus

- gpt-5-nano

- ...

我们以qwen3.5-plus为例，演示向模型发送图片消息

##### 3.2.1 在线图片

首先，我们演示如何发送一个在线图片给模型，也就是指定模型的url地址。

图片如下：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/IfzIbKohJoR43WxmjAhc0xmwnef/)

消息格式如下：

```JSON
{
```

示例代码：

```Python
from langchain.chat_models import init_chat_model
import os
```

结果：

```Plain Text

```

##### 3.2.2 本地图片

所谓本地图片，就是用户上传的图片数据或者本地存在的图片，而不是图片的url地址。我们需要将图片数据转换成base64字符串，然后发送给模型。

本地图片的消息格式：

```JSON
{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}
```

示例：

```Python
import base64

# 例如，有一个用户上传的文件，是字节格式img_bytes，我们先将其进行base64编码
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 组织多模态消息
multimodal_question = HumanMessage(content=[
    {
        "type": "image",
        "base64": img_b64,
        "mime_type": "image/jpeg",
```

### 4. 提示词（Prompts）

发送给大模型的所有消息都可以称为**提示词（Prompt）**，它直接影响模型的输出结果**。**

其中，SystemMessage尤为重要，我们把SystemMessage称为**系统提示词**（**System Prompt**），它可以给模型设定角色和本次聊天的背景，对模型生成的内容有很大的影响。

#### 4.1 系统提示词

在创建智能体时，我们可以直接设定system prompt，不必在每次发送消息时指定。

```Python

```

如果没有设定系统提示词，模型会按照训练中的自我认知来回答：

```Markdown
你好！我是DeepSeek，由深度求索公司创造的AI助手！

我是一个纯文本模型，虽然不支持多模态识别功能，但我有文件上传功能，可以帮你处理图像、txt、pdf、ppt、word、excel等各种文件，从中读取文字信息进行分析处理。

我的一些特点：
- 完全免费使用，没有收费计划
- 上下文长度达128K，可以处理很长的对话
- 支持联网搜索（需要你手动点开联网搜索按键）
- 可以通过官方应用商店下载App使用
```

而设定了**海盗**这个角色后，它的回答就非常有趣了：

```Plain Text
啊哈！我是你船上的鹦鹉，一个在数字海洋里翱翔的AI助手！想聊聊宝藏、航海，还是七大洋的奇闻？尽管放马过来，伙计！
```

#### 4.2 提示词工程

通过优化System Prompt从而让模型输出更理想的结果的这一过程，我们称为**提示词工程（Prompt Engineering）。**

也就是说，提示词优化不是一锤子买卖，而是一个不断优化、测试、再优化的过程。那么，提示词到底该怎么写呢？

从**内容**来说，提示词通常包含以下几个部分，通常按此顺序排列：

- **身份（Identity）**：描述AI的职责、沟通风格和总体目标。

- **说明（Instructions）**：请指导模型如何生成所需的响应。它应该遵循哪些规则？模型应该做什么，以及模型绝对不能做什么？

- **示例（Examples）**：提供可能的输入示例，以及模型期望的输出。

- **背景信息（Context）**：向模型提供生成响应所需的任何额外信息，例如RAG的额外知识库数据，或您认为特别相关的任何其他数据。

从**格式**来说，在编写System Prompt时，您可以使用Markdown格式和XML 标签的组合来帮助模型理解提示和上下文数据的逻辑边界。

- **Markdown** 的标题和列表有助于标记提示的不同部分，并向模型传达层级结构。它们还可以提高开发过程中提示的可读性。

- **XML** 标签可以帮助明确区分一段内容（例如用作参考的辅助文档、对话示例等）的起始和结束位置。

示例：

```Markdown
# Identity

You are a helpful assistant that labels short product reviews as
Positive, Negative, or Neutral.
```

接下来，我们就学习下不同的提示词对模型结果的影响。

##### 4.2.1 设定角色和详细指令

**角色**可以帮助模型认清自己的身份，以对应的身份来回答问题。

**指令**则告诉模型需要遵循哪些规则，应该做什么，不应该做什么

例如：

```Python

```

输出结果：

```Python

```

##### 4.2.2 Few-Shot examples

有的时候我们希望模型按照固定的风格来回答问题，而这种风格又不太好描述，那我们就可以通过举例的方式让模型学习例子来回答。

用户只需在输入提示（Prompt）中提供几个输入-输出示例，模型就能理解任务模式并生成符合预期的输出：

```Python

system_prompt = """
# 身份
- 你是一个科幻作家，根据用户的要求创建一个太空之都。

# 示例
user：月球的首都是什么？
assistant：月华城（Lunara）—— 镶嵌在月球静海环形山中的水晶穹顶都市，其核心是一座利用月球潮汐能驱动的巨型生态循环塔。
```

结果：

```Python
熔金城（Aurum）——悬浮于硫酸云层之上的宏伟浮空都市，以反光性合金铸造，永恒折射着昏黄的日光。
```

##### 4.2.3 结构化输出

由于传统程序识别结构化的数据会更加方便，所以有时候我们希望LLM也能输出固定结构的内容，方便我们解析。这同样可以通过系统提示词来实现。

```Python

system_prompt = """
# 身份
- 你是一个科幻作家，根据用户的要求创建一个太空之都。

# 指令
- 请务必以JSON格式输出，不要加任何markdown样式。

# 示例：
user: 月球的首都是什么？
assistant:
{
    "name": "月华市（Lunaria）",
    "location": "位于月球正面赤道附近的静海基地遗址之上，依托巨大的穹顶与地下网络建成",
```

输出结果：

```JSON
{
    "name": "硫磺城（Sulfura）",
    "location": "悬浮于金星浓厚大气层中距地表约50公里的高空，由巨大的反重力浮空平台群构成",
    "vibe": "高压、炽热、坚韧",
    "economy": "大气资源提炼（二氧化碳、硫酸）、极端环境材料制造、太阳能巨型阵列"
}
```

在LangChain中，实现结构化输出会更加简单。我们无需自己在提示词中添加描述实现结构化输出，而仅仅是设定好一个数据类型即可。

首先，我们定义一个类，用来封装模型要输出的数据：

```Python

```

然后，我们就可以在创建Agent时设定好输出格式：

```Python

```

注意，在输出的结果中，有一个'structured_response'的字段，就是结构化输出的对象：

```Python

```

所以，我们这样获取结构化的输出：

```Python
city = response['structured_response']
```

完整代码：

```Python

```

#### 4.3 总结

本节我们主要学习了：

- 什么是提示词

- 如何优化系统提示词，控制模型输出

仔细回忆一下，以上知识你都掌握了吗？

到目前为止，我们的Agent与普通的聊天机器人没什么差别，只能实现简单的问答。

这是因为，它还缺少一个非常重要的东西:Tools

下一节，我们就正式学习Tool

### 5. 工具（Tools）

一个完整的Agent至少要包含两个关键的部分：

- **模型**：是Agent的大脑，负责推理、分析，规划任务步骤

- **工具**：是Agent的手脚，负责执行任务，与外界交互

> [白板/画板内容] (token: Fq5odjkRwoPbbmxpl4tcISfUnZf)

#### 5.1 基本用法

我们先通过一个案例快速回顾Agent定义的步骤，以及Agent的工作原理。

定义一个带有工具的Agent分为两步：

- 定义工具

- 定义Agent，绑定工具

首先，使用tool装饰器定义工具：

```Python
# 1.使用tool装饰器定义工具
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """
    Get the weather in a given location.
```

接着，定义Agent，绑定工具：

```Python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 2.创建智能体，并绑定工具
agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather]
```

执行结果如下：

```Python
================================ Human Message =================================
```

流程图：

> [白板/画板内容] (token: BfsbdyZASovmaCxitoTcgDEsnUf)

由此可见，所谓的工具，本质就是一个**可调用的函数**，要想让Agent知道有哪些工具可调用，该如何调用这些工具，就必须把这个函数的详细信息发送给模型。包括：

- 函数名

- 函数的作用

- 函数的参数和返回值信息

所以，定义工具的时候，关键就是把这些信息描述清楚即可。

#### 5.2 自定义工具

在LangChain中，定义工具的过程被大大简化，与定义普通函数几乎没什么差别，只是在一些细节上需要注意。

首先，定义工具需要在函数上添加`@tool`装饰器。

例如，我们定义一个计算平方根的数学工具：

```Python
# 定义工具
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """计算指定数字的平方根"""
    return x ** 0.5
```

智能体在工作时，需要将函数的名称、输入、作用传递给大模型，默认情况下这些信息的来源是：

- 工具名称：函数名

- 工具输入：函数入参

- 工具作用：函数的注释

当然，我们可以通过tool装饰器来覆盖上述信息：

- 通过装饰器定义工具名称

```Python
@tool("square_root")
def tool1(x: float) -> float:
```

- 通过装饰器定义工具作用描述

```Python

```

- 通过装饰器定义工具入参约束

如果要覆盖工具的入参信息则会复杂很多，我们要借助于Pydantic或JSON约束。

例如，我们需要定义个查询天气的tool，借助于Pydantic来约束入参。

我们定义一个入参的模型，在模型中添加入参描述信息：

```Python
# 例如一个查询天气的tool
class WeatherInput(BaseModel):
    """查询天气的输入参数."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
```

定义工具，使用定义的模型来约束入参：

```Python
# 定义一个查询天气的tool
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
```

工具定义好之后，调用方式与普通函数类似：

```Bash
# 调用数学工具
tool1.invoke({"x": 467})

# 调用查询天气工具
get_weather.invoke({"location": "杭州", "include_forecast": True})
```

 
> **注意**：
> 在LangChain中，作为工具的函数**有两个保留的参数名**，你的自定义参数不能与之重复！他们是：
>   - **config**：用来传递运行时配置
>   - **runtime**：用来传递运行时上下文

当我们创建智能体时，可以把定义好的工具传递给智能体，将来模型就能得到工具信息，并根据情况判断是否需要调用工具，需要调用哪个工具了。

```Bash
from langchain.agents import create_agent

# 创建智能体，并添加工具
agent = create_agent(
    model="deepseek-chat",
    tools=[tool1, get_weather],
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。"
)
```

接下来，调用智能体，向其提问，模型会自动根据用户问题判断：

- 是否需要调用工具？

- 该调用哪个工具？

- 该传递那些参数？

并且在调用工具之后，根据工具执行结果给用户生成响应。

```Python

```

如果采用stream模式的updates模式，可以看到工具调用的具体步骤：

```Python
for chunk in agent.stream(
    {"messages": [HumanMessage(content="467、529的平方根是多少?")]},
```

输出如下：

```SQL
step: model
content: [{'type': 'text', 'text': '我来帮你计算这两个数的平方根。'}, {'type': 'tool_call', 'id': 'call_00_oWChR8Xgo21mmWKW0SP9uOS9', 'name': 'square_root', 'args': {'x': 467}}, {'type': 'tool_call', 'id': 'call_01_UqzhGeRNcoSoidItA0gScaoY', 'name': 'square_root', 'args': {'x': 529}}]
```

工作流程如图：

> [白板/画板内容] (token: XgdTdzCvWoXGn2xp57ucGB1Tnik)

#### 5.3 预定义工具

LangChain中提供了很多预定义好的工具，方便我们使用，可使用的预定义工具列表可参考官网：

[Tool integrations - Docs by LangChain](https://docs.langchain.com/oss/python/integrations/tools)

例如，模型本身只能根据本身的训练数据回答问题，无法获取实时信息。但如果我们给它提供了web搜索的工具，那么你的Agent就如同具备了实时web搜索的能力，回答会更加准确。

有一个专门用于给Agent提供Web搜索的工具，叫做Tavily，官网如下：

[嵌入网页](https://www.tavily.com/)

在LangChain中也提供了对Tavily的支持：

[Tavily search integration - Docs by LangChain](https://docs.langchain.com/oss/python/integrations/tools/tavily_search)

要使用这个工具，步骤如下：

##### 5.3.1 注册账号

首先，我们要在Tavily官网注册一个账号，可以选择邮箱注册，或者直接用google、github登录：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BnewbY0Trohdz7xzKUecnD9jnyb/)

注册成功后，我们登录后台（[https://app.tavily.com/home](https://app.tavily.com/home)），即可看到一个默认的API_KEY：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/OCZebP1gyo5kv5xSi8Bcqg53nyg/)

##### 5.3.2 配置环境变量

接下来，我们需要把这个KEY配置到我们的.env文件中：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/INo7bgrq1oWhScxHRAAcYLBfnYd/)

##### 5.3.3 安装依赖

然后，我们需要安装langchain-tavily的依赖：

```Bash
# 使用uv的环境
uv add langchain-tavily
```

##### 5.3.4 使用工具

接下来，就可以使用tavily来做web搜索了：

```Python
# 使用tavily作为web搜索工具
```

这里我们无需自己定义工具，而是直接导入即可，然后根据自己的需要设置参数。

完整的参数设置参考官网：[langchain官网参考](https://docs.langchain.com/oss/python/integrations/tools/tavily_search)、[tavily官网参考](https://docs.tavily.com/documentation/api-reference/endpoint/search)

试试看调用工具：

```Bash
tool.invoke("杭州今天天气如何？")
```

结果：

```SQL

```

##### 5.3.5 结合智能体

我们结合智能体来使用Tavily搜索工具：

```Bash
# 创建智能体，使用预定义工具tavily
agent = create_agent(
    model="deepseek-chat",
    tools=[tool],
    system_prompt="你是一个智能助手，你使用工具来解决用户问题。"
```

执行过程：

```JSON
step: model
content: [{'type': 'text', 'text': '我来帮您查询北京接下来5天的天气情况。'}, {'type': 'tool_call', 'id': 'call_00_e0PbwzrTI77ojtewzw9VuPfR', 'name': 'tavily_search', 'args': {'query': '北京未来5天天气预报', 'search_depth': 'basic', 'time_range': 'day'}}]

step: tools
content: [{'type': 'text', 'text': '{"query": "北京未来5天天气预报", "follow_up_questions": null, "answer": null, "images": [], "results": [{"url": "https://www.bjfsh.gov.cn/zhxw/fsdt/202602/t20260228_40112151.shtml", "title": "未来一周气温“过山车”！具体预报—— - 房山区", "content": "访问我的专属空间 公务员邮箱 繁体; \\"繁体\\") 简体; \\"简体\\") 无障碍 长者专区. 日期:2026-02-28 09:47         来源:北京发布微信公众号 气象北京微信公众号. 北京市气象台27日14时发布：今天下午阴转多云，偏南风二三级，**最高气温7℃；**夜间多云转阴，南转东风一二级，**最低气温-1℃。**今天气温下降，外出注意防寒保暖。. 雨雪过后，部分路面湿滑，**目前平谷、房山、密云、门头沟、昌平、延庆、怀柔区仍处于道路结冰黄色预警中，**提醒大家出行注意安全，驾车减速慢行，保持车距。. 未来三天，我市低层湿度较大，天气以多云到阴天为主，气温较低。**明天白天最高气温将降至4℃左右，西部北部有雨夹雪，**户外体感阴冷，大家外出要多穿一点，以免着凉感冒。. **3日（元宵节）白天最高气温将升至10℃左右，**大部分时间比较适宜大家出行，夜间最低气温-1℃，户外活动需注意保暖。下周冷空气活动频繁，天空云量较多，气温波动起伏，请关注临近预报，根据天气和气温变化，适时调整着装。. **27日（星期五）下午：**阴转多云；偏南风2、3级；**平原地区最高气温7℃，**山区最高气温5～7℃；最小相对湿度40%。. **27日夜间：**多云转阴；南转东风1、2级；平原地区最低气温-1℃，山区最低气温-4～-3℃；最大相对湿度85%。. **28日（星期六）白天：**阴，**西部、北部有雨夹雪；**东转南风2、3级；**平原地区最高气温4℃，**山区最高气温2～4℃；最小相对湿度50%。. **28日夜间：**阴，**山区有零星小雪；**南转北风1、2级；平原地区最低气温-1℃，山区最低气温-5～-2℃。. **1日（星期日）白天：**阴转多云；北转南风2、3级；**平原地区最高气温5℃**，山区最高气温4～6℃。. **1日夜间：**多云转阴；南转北风1、2级；平原地区最低气温0℃，山区最低气温-3～-1℃。. * 客服信箱：webmaster@bjfsh.gov.cn.", "score": 0.72231215, "raw_content": null}, {"url": "https://news.bjd.com.cn/2026/02/28/11605674.shtml", "title": "未来三天北京多弱雨雪，4日夜间至5日白天将有降雪 - 新闻- 京报网", "content": "热   点 锐   评 发布厅 城   事 影   视 视   觉 京   剧 汽   车 纸上听 时   事 学   习 都视频 艺   绽 深   读 京   味 体   育 天   下 长   城 经   济. 热点 锐评 城事 影视 辟谣 京剧 都视频 电子报 汽车 时事 学习 视觉 艺绽 深读 京味 纸上听 体育 天下 长城 经济 北京民声 北晚在线. 今天白天北京天空阴沉，有弱雨雪天气，最高气温仅为2℃左右，体感阴冷，外出注意防寒保暖和交通安全。夜间山区仍有零星小雪，最低气温在冰点以下。. 阴，平原地区有雨夹雪，山区有小雪；偏东风2级；2～1℃。. 阴；偏北风1、2级；-1～1℃。. 28日（周六）下午：阴，平原地区有雨夹雪，山区有小雪；偏东风2级左右；平原地区最高气温2℃，山区最高气温0～2℃；最小相对湿度50%。. 28日夜间：阴，山区有零星小雪；东转北风1、2级；平原地区最低气温-1℃，山区最低气温-5～-4℃；最大相对湿度70%。. 3月1日（周日）白天：阴转多云；北转南风2、3级；平原地区最高气温6℃，山区最高气温5～7℃；最小相对湿度50%。. 3月1日夜间：多云转阴，大部地区有雨夹雪或小雪；南转北风1、2级；平原地区最低气温0℃，山区最低气温-2～-1℃。. 2日（周一）白天：阴，大部地区有雨夹雪或零星小雪；北转南风2、3级；平原地区最高气温6℃，山区最高气温2～4℃。. 2日夜间：阴，山区有小雪；南转北风1、2级；平原地区最低气温-1℃，山区最低气温-2～-1℃。. 明天北京整体还是阴到多云为主，湿度较大，白天最高气温6℃左右。3月1日夜间到2日白天大部地区有雨夹雪或小雪天气，雨雪将导致路面湿滑、能见度下降，对2日（周一）早高峰有不利影响，外出注意防范雨雪和行车安全。. 3月1日至8日气温变化幅度较大，最高气温3~10℃，最低气温-4~1℃。. 如遇作品内容、版权等问题，请在相关文章刊发之日起30日内与本网联系。版权侵权联系电话：010-85202353. 人民网 新华网 央视网 光明网 中国网 中国日报网 中国经济网 千龙网 今日头条 百度 新浪 网易 腾讯 搜狐 爱奇艺 优酷. 主管：北京日报报业集团 主办：京报移动传媒有限公司 Copyright ©1996-2026 Beijing Daily Group, All Rights Reserved. 互联网新闻信息服务许可证11120180001号 网络出版服务许可证（京）字第338号 信息网络传播视听节目许可证0122682号 广播电视节目制作经营许可证（京）字第00693号. 京公网安备11040202440037号  工信部备案号：京ICP备16035741号-1 京新网备2010001号   网上有害信息举报专区    北京互联网举报中心. Copyright ©1996-2026 Beijing Daily Group, All Rights Reserved.", "score": 0.64478505, "raw_content": null}, {"url": "https://cj.sina.com.cn/articles/view/1617264814/606580ae02002jbfe", "title": "未来三天北京多弱雨雪，4日夜间至5日白天将有降雪 - 新浪财经", "content": "外汇 管理 消费 科技 互联网 手机. 保险 数码 科普 创业 银行 新三板 其他. # 未来三天北京多弱雨雪，4日夜间至5日白天将有降雪. 语音播报 缩小字体 放大字体 微博 微信 分享 0. 今天白天北京天空阴沉，有弱雨雪天气，最高气温仅为2℃左右，体感阴冷，外出注意防寒保暖和交通安全。夜间山区仍有零星小雪，最低气温在冰点以下。. 阴，平原地区有雨夹雪，山区有小雪；偏东风2级；2～1℃。. 阴；偏北风1、2级；-1～1℃。. 28日（周六）下午：阴，平原地区有雨夹雪，山区有小雪；偏东风2级左右；平原地区最高气温2℃，山区最高气温0～2℃；最小相对湿度50%。. 28日夜间：阴，山区有零星小雪；东转北风1、2级；平原地区最低气温-1℃，山区最低气温-5～-4℃；最大相对湿度70%。. 3月1日（周日）白天：阴转多云；北转南风2、3级；平原地区最高气温6℃，山区最高气温5～7℃；最小相对湿度50%。. 3月1日夜间：多云转阴，大部地区有雨夹雪或小雪；南转北风1、2级；平原地区最低气温0℃，山区最低气温-2～-1℃。. 2日（周一）白天：阴，大部地区有雨夹雪或零星小雪；北转南风2、3级；平原地区最高气温6℃，山区最高气温2～4℃。. 2日夜间：阴，山区有小雪；南转北风1、2级；平原地区最低气温-1℃，山区最低气温-2～-1℃。. 明天北京整体还是阴到多云为主，湿度较大，白天最高气温6℃左右。3月1日夜间到2日白天大部地区有雨夹雪或小雪天气，雨雪将导致路面湿滑、能见度下降，对2日（周一）早高峰有不利影响，外出注意防范雨雪和人员行车安全。. 3月1日至8日气温变化幅度较大，最高气温3~10℃，最低气温-4~1℃。. ## 头条号入驻. ## 财经自媒体联盟更多自媒体作者. ## 热文排行榜. * 04 在美国，我感受到“越南制造”在取代“中国制造”. * 06 三星“宫心计”：儿子检举父亲，父母逼死女儿. * 10 56岁创业，年利是华为1.5倍，他是让对手发抖的人！. 关于头条 | 如何入驻 | 发稿平台 | 奖励机制 版权声明 | 用户协议 | 帮助中心. Copyright © 1996-2019 SINA Corporation. All Rights Reserved 新浪公司 版权所有.", "score": 0.61307496, "raw_content": null}, {\"url\": \"https://www.nmc.cn/f/index-1.html\", \"title\": \"关注阴晴冷暖，守望四时安康 - 中央气象台\", \"content\": \"北京 18.3℃ 西南风 微风. 北方冬麦区气象条件利于小麦越冬 5-7日中东部雨雪降温影响设施农业生产 未来10天江南西部和西南地区东部多阴雨 须注意防范湿渍害. [2026年2月9日[新闻直播间]中央气象台\\\\_中东部气温波动回升\\\\_累积升幅可超10℃](/publish/cms/view/c5e4387dc33441e4853dead0d400b0b9.html). [2026年2月9日[新闻直播间]中央气象台\\\\_江南等地雨雪再度发展\\\\_春运出行需防范](/publish/cms/view/5db2a19c37264a97a634ab5afd39ed17.html). [2026年2月6日[新闻直播间]中央气象台\\\\_寒潮影响持续\\\\_江南局地降温将超12℃](/publish/cms/view/6fb5e2802bc5421480f4b5eb463ff0d6.html). #### 城市定制. 关于我们 联系方式 网站声明 业务信息 网站动态 网站地图 触摸屏版. 国家气象中心 版权所有 Copyright©2009-2026. 制作维护：国家气象中心预报系统开放实验室 地址：北京市中关村南大街46号 邮编：100081.\", \"score\": 0.57236487, \"raw_content\": null}, {\"url\": \"https://www.qweather.com/weather30d/beijing-101010100.html\", \"title\": \"北京市未来30天天气预报\", \"content\": \"北京市. 中国. 2026-02-28. 未来30天将有3天下雪，最高温20°（03月13日，03月14日），最低温-4°（03月05日）。 日. 一. 二. 三. 四. 五. 六. 28 QWeather. 2°~0°.\"", \"score\": 0.5125224, \"raw_content\": null}], \"response_time\": 0.8, \"request_id\": \"1809e393-2e22-48c9-bed9-33698921890a\"}'}]\n```\n\n##### 5.3.6.优化\n\n注意，LangChain提供的TavilySearch工具描述非常复杂，参数也很多。会有额外的网络消耗。如果我们仅仅是需要query参数，建议自定义工具。\n\n像这样：\n\n```Python\n# 使用tavily作为web搜索工具\ntavily = TavilySearch(\n    max_results=5,\n    topic=\"general\"\n```\n\n默认情况下AI回答的结果不包含信息来源，这样回答的可信度就不高。我们可以自定义结构化输出，让AI在回答时包含信息来源。\n\n```Python\n\n```\n\n结果如下：\n\n```Plain Text\nanswer='\"蒸蚌\"是一个网络谐音梗，意思是\"真棒\"。\\n\\n**具体含义：**\\n- \"蒸蚌\"是\"真棒\"的谐音，因为这两个词在普通话中发音相似（zhēn bàng）\\n- 这是一种网络幽默用法，故意用\"蒸蚌\"这两个看起来不相关的字来代替\"真棒\"，制造一种有趣、搞笑的效果\\n\\n**使用场景：**\\n1. 在社交媒体、聊天中用来表达赞美、鼓励\\n2. 常用于宠物视频中，比如抖音上流行的\"萝卜纸巾挑战\"中，当宠物选对时主人会喊\"蒸蚌！\"\\n3. 作为一种网络幽默表达方式，增加趣味性\\n\\n**来源和流行：**\\n这个梗最初可能源于网络聊天中的谐音创造，后来在抖音等短视频平台上因为\"萝卜纸巾挑战\"而流行起来。在这个挑战中，宠物需要在胡萝卜和纸巾之间做出选择，选对了就会得到主人\"蒸蚌！\"的夸奖。\\n\\n**类似梗：**\\n类似的谐音梗还有\"蒸虾\"（真瞎）、\"蚌埠住了\"（绷不住了）等，都是利用汉字谐音创造幽默效果的网络用语。' \n```\n\n### 6. 记忆（memory）\n\n模型本身是没有记忆的，它记不住历史的会话内容，参考之前的章节介绍：[第1章. AI通识与基础](https://my.feishu.cn/wiki/PAb6wSNnziRrlEk1PtNcH38LnJZ?fromScene=spaceOverview#share-I3mBdWHvDoFsV5xqvuvc56MNnZe)\n\n我们需要通过技术手段，帮助模型记住会话历史，产生记忆。\n\n对于Agent而言，记忆至关重要，因为它能让代理记住之前的交互情况，从反馈中学习，并适应用户的偏好。随着代理处理的任务愈发复杂，涉及的用户交互也越来越多，这种能力对于提高效率和用户满意度而言变得不可或缺。\n\n#### 6.1 记忆的分类\n\n对于智能体而言，记忆分为了两类：\n\n- 短期记忆（short-term memory）\n\n- 长期记忆（long-term memory)\n\n注意，大家不要被字面上的意思误导了，很多人看到名字就误以为：*短期记忆就是临时记忆，断电就没了；长期记忆就是永久记忆，持久保存*。\n\n对于智能体而言，这是完全错误的理解！！！\n\n简单用一句话概括的话：\n\n- **短期记忆**：当前任务或会话的上下文（Working Memory 或 Session Memory）\n\n- **长期记忆**：跨任务或会话的**经验与知识**（Persistent Memory）\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Ozznb7xjNoqbQbx9m7Scwe4jn0g/)\n\n比如，一个公司数据分析的Agent。\n\n用户提出需求：\n\n> “帮我写Q1的销售分析报告”\n\nAgent：\n\n---\n\n短期记忆：\n\n- 对话历史\n\n- 查询到Q1的销售数据\n\n- 任务目标及执行状态\n\n---\n\n长期记忆：\n\n- 公司的KPI算法\n\n- 用户偏好的报告形式\n\n总结：\n\n|  | 短期记忆 | 长期记忆 |\n| --- | --- | --- |\n| 生命周期 | 当前会话（短暂） | 跨任务、跨会话（永久） |\n| 内容 | 当前任务状态 | 知识、经验、用户偏好 |\n| 是否跨任务 |  |  |\n|  |  |  |\n\n接下来，我们先学习LangChain中的短期记忆管理。\n\n#### 6.2 短期记忆\n\n由于**短期记忆**通常生命周期是当前会话，所以我们也可以称为**会话记忆**。Agent的会话记忆通常包含三部分：\n\n- 对话历史\n\n- 查询结果\n\n- 任务状态\n\n对于简单的Agent来说，任务没有做拆分，也就不需要记录任务状态，只用考虑**会话历史**和**查询结果**就可以了。后续我们会学习如何自定义更复杂的Agent会话记忆。\n\nLangChain提供了自动化的记忆管理方案：\n\n- 首先，LangChain把会话记忆（也就是Messages列表）记录为**AgentState**的一部分\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/NOgBbMYw3o1IHxxJAIzcf98GnSf/)\n\n- AgentState通过**Checkpointer**对象来保存，每一次与AI的交互都会生成一个快照，记录为一个StateSnapshot，把同一会话的所有StateSnapshot组合在一起，就是完整的会话历史了。\n\n- 为了区分不同的会话记忆，不同会话需要设定各自的`thread_id`，相同会话则使用相同`thread_id`\n\n- 向Agent发起会话时必须指定自己的`thread_id`以唤起对应的会话记忆\n\n> [白板/画板内容] (token: OiizdgnT2oBYN3xD4hNcuSn9nGc)\n\n接下来，我们以LangChain提供的基于内存的Checkpointer为例来演示会话记忆。\n\n##### 6.2.1 InMemorySaver\n\n具体步骤是这样的：\n\n1. 导入CheckPointer的内存版实现：\n\n1. 创建智能体，设置checkpointer：\n\n1. 发起调用时，指定thread_id\n\n由于两次调用使用了相同的thread_id，被认定为是同一次对话，所以LangChain会在请求模型时携带历史对话的Messages，模型就能根据历史消息来正确回答了：\n\n \n> **注意**：\n\n##### 6.2.2 持久化Memory（选学）\n\nLangChain也提供了很多持久化存储的checkpointer，例如：\n\n- SqlLiteSaver ：基于sqlite存储\n\n- PostgresSaver ：基于Postgres存储\n\n- CosmosDBSaver ：使用Azure Cosmos DB的实现\n\n我们以SqlLiteSaver 为例来讲解如何自定义Memory存储方案。\n\n**首先**，安装对应以来：\n\n```Bash\n\n```\n\n**然后**，导入以来，并初始化sqlite-checkpointer\n\n```Python\nimport sqlite3\nfrom langgraph.checkpoint.sqlite import SqliteSaver\n```\n\n**最后**，创建Agent，并设置checkpointer：\n\n```Bash\n# 创建agent\nagent = create_agent(\n    \"deepseek-chat\",\n    checkpointer=checkpointer,\n)\n```\n\n#### 6.3 记忆管理策略\n\n由于会话记忆要保存会话的历史，并且在调用LLM时携带历史消息列表。而当会话越来越长时，历史消息就可能超过LLM的上下文限制。例如，DeepSeek的上下文不能超过128K.\n\n一旦会话历史超过上下文窗口，就会出现上下文丢失的情况，从而导致丢失记忆。而且即便不丢失，太长的上下文容易让模型出现“注意力分散”问题，模型的响应速度、回答质量会大大降低。\n\n未来解决这一问题，通常有以下几种手段：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/T5Gibx6YuoAkkZxAgalcR6nxnDf/)\n\n具体可参考官网：\n\n[Short-term memory - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory#common-patterns)\n\n##### 6.3.1 修剪消息\n\n修剪消息并不是真正的删除消息，在AgentState中的消息列表依然是完整的，只不过发送给LLM之前会进行修剪，只保留一部分消息。\n\n> [白板/画板内容] (token: EbPTdR64ToipYsxNm7icWdxCnPf)\n\n具体示例参考：\n\n[Short-term memory - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory#trim-messages)\n\n##### 6.3.2 删除消息\n\n删除消息与修剪不同：\n\n- 修剪消息：只是从State中选取一部分消息发送给模型\n\n- 删除消息：直接删除State中保存的消息，也就是说消息历史中不再存在！\n\n所以一定要谨慎使用。\n\n具体参考：\n\n[Short-term memory - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/short-term-memory#delete-messages)\n\n##### 6.3.3 总结消息\n\n不管是修剪还是删除，都会导致一部分消息丢失，从而丢失记忆。所以就有了第三种策略：**总结消息（Summarize Messages）**\n\n它的思路很简单，就是把历史的消息利用大模型总结出摘要，然后把最新的消息拼接在一起作为新的消息列表发送给大模型，这样既不会超出模型的上下文窗口限制，还能尽量保留所有的记忆。\n\n> [白板/画板内容] (token: NtjudOEVGotnXZxCDZgcU0vgnfe)\n\nLangChain提供了总结消息的默认实现：**SummarizationMiddleware**\n\n用法很简单：\n\n1. 初始化SummarizationMiddleware和checkpointer\n\n1. 创建Agent，设置middleware和checkpointer\n\n1. 调用Agent即可\n\n测试结果：\n\n```Python\n================================ Human Message =================================\n\nHere is a summary of the conversation to date:\n```\n\n---\n\n## 第2节. Agent入门实战（AI私厨）\n\n本章我们要利用前面所学的知识实现一个多模态智能体应用：AI私厨。\n\n### 1. 需求分析\n\nAI私厨是一个基于LangChain和多模态AI的食谱推荐应用。用户可以拍摄自家冰箱或厨房的食物照片，AI会自动识别图片中的食材，根据食材搜索相关食谱推荐给用户。\n\n#### 1.1 功能特性\n\n- 图片识别 - 上传食材图片，自动识别其中的食材\n\n- 智能搜索 - 根据识别的食材搜索相关食谱\n\n- 智能排序 - 按推荐度、难度、营养价值对食谱进行排序\n\n- 创意建议 - 当找不到合适食谱时，提供创意搭配建议\n\n- 对话交互 - 聊天式界面，支持图片上传 + 文本对话\n\n#### 1.2 预期效果\n\n基本的聊天界面窗口，支持**图片上传**+**文本对话**：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/R6A8bHXRroxYtmxtyBAcdpnHnvd/)\n\n上传图片，根据图片识别食材：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BMHDbd9SToTEJZxHnZYc1vwandG/)\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Xu2Yb8AKHoiUELx0Cn0csDjVnXf/)\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/XapqbzsNXogHoaxsgYqcr0bcnNh/)\n\n根据食材搜索食谱，并智能排序：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/R45ubBHFHo5dOvx2W3Jcl3f3nPc/)\n\n#### 1.3 实现思路分析\n\n定义一个基础Agent，核心就是模型（Model）和工具（Tool），我们从这两点入手：\n\n- **模型**：由于要实现多模态输入，所以必须用多模态模型，比如qwen3.5-plus\n\n- **工具**：需要联网搜索食谱，所以要用到web搜索工具，推荐Tavily\n\n- **记忆**：短期记忆我们暂时使用Sqlite来实现\n\n剩下就是提示词了，这里我给大家准备了一份测试过的提示词：\n\n```Python\nsystem_prompt = \"\"\"\n你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：\n1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。\n2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。\n3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。\n4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。\n```\n\n接下来，就是组合这些东西，开发你的智能体了，赶紧动手试试吧！\n\n### 2. 功能模拟（jupyter）\n\n我们先在jupyter将Agent的流程跑通。\n\n#### 2.1 配置\n\n确保你的.env中有以下配置：\n\n```Properties\n# .env\n\n# 阿里百炼\nDASHSCOPE_API_KEY=sk-913a82aa121f412aa9a8c8c7b22f7792\nDASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n\n# web_search引擎\nTAVILY_API_KEY=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\n```\n\n#### 2.2 依赖\n\n参考下面的依赖：\n\n```TOML\n[project]\nname = \"food-recipe-recommender\"\nversion = \"0.1.0\"\ndescription = \"AI-powered recipe recommender based on uploaded food images\"\nreadme = \"README.md\"\nrequires-python = \">=3.10\"\ndependencies = [\n    \"langchain>=0.3.0\",\n```\n\n#### 2.3 加载配置\n\n首先，我们要加载环境变量：\n\n```Python\n# 加载环境变量\nfrom dotenv import load_dotenv\n```\n\n#### 2.4 定义工具\n\n然后，我们要定义工具：\n\n```Python\nfrom langchain_tavily import TavilySearch\n\n# web搜索工具，使用tavily作为web搜索工具\nweb_search = TavilySearch(\n    max_results=5,\n    topic=\"general\",\n```\n\n#### 2.5 初始化模型\n\n接着，初始化多模态模型：\n\n```Python\nfrom langchain.chat_models import init_chat_model\nimport os\n\nmodel = init_chat_model(\n    model=\"qwen3.5-plus\",  # 模型名称，这里选择qwen3.5-plus，这是一个多模态模型\n    model_provider=\"openai\",\n```\n\n#### 2.6 记忆管理\n\n然后，我们定义记忆管理的checkpointer:\n\n```Python\nfrom langgraph.checkpoint.sqlite import SqliteSaver\nimport sqlite3\n```\n\n#### 2.7 初始化Agent\n\n接下来，就是智能体了：\n\n```Python\nfrom langchain.agents import create_agent\n```\n\n#### 2.8 测试\n\n我们在网上找到一张冰箱食物图片：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/MTv7belxuoW1wCxOsg1cJyuNnkh/)\n\n测试一下：\n\n```Python\nfrom langchain.messages import HumanMessage\n\n# 准备多模态消息，图片是网络上的冰箱食物图片\nmultimodal_message = HumanMessage(\n    content=[\n        {\"type\": \"image\",\n```\n\n打印：\n\n```Python\n# 友好打印\nfor message in response['messages']:\n    message.pretty_print()\n```\n\n结果：\n\n```SQL\n================================ Human Message =================================\n\n[{'type': 'image', 'url': 'https://img.freepik.com/free-photo/arrangement-different-foods-organized-fridge_23-2149099882.jpg'}, {'type': 'text', 'text': '帮我看看这些食材能做些什么？'}]\n================================== Ai Message ==================================\n\n这些食材非常丰富，包括了多种蔬菜、菌菇、鱼类和肉类，非常适合制作健康美味的餐食。我将根据这些食材，为您搜索可用的食谱，并提供创意搭配建议。同时，我会对每个食谱的营养价值和制作难度进行评分，并综合排序，为您提供最佳选择。\n```\n\n继续询问：\n\n```Python\n\n```\n\n结果：\n\n```Markdown\n================================== Ai Message ==================================\n```\n\n可以发现，Agent具备记忆，接着前面的内容来回答。\n\n测试通过！\n\n### 3. LangSmith联调测试\n\nLangChain的Agent底层是基于LangGraph实现的，而LangGraph提供了完整的后端部署功能，自带非常完善的API接口，无需我们额外处理。\n\n同时，LangChain还提供了基于LangSmith的GUI控制台实现Agent的调试、监控、一键部署。\n\n接下来我们就看看如何利用LangGraph在本地部署测试我们的Agent，并通过LangSmith做测试。\n\n#### 3.1 配置LangSmith\n\nLangSmith提供了对Agent的GUI管理界面，而且还支持一键云部署功能。通常在测试阶段，建议大家在Agent中引入Simth，方便做测试和调试。\n\n首先，我们要注册LangSmith，开通服务，生成API_KEY。\n\n注册地址：\n\n[LangSmith](https://smith.langchain.com/)\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/AXvgbnOZzo7BhixpAY8cvoPQnrd/)\n\n注册成功后，登录，进入控制台，找到settings：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/SUyDbSBLfofSBbxjR4LcrTMOnGg/)\n\n在settings页面找到API Keys菜单，创建自己的API_KEY：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/UMtUbp9RAok5WgxSrEhcbDfln4V/)\n\n千万要记住自己的API_KEY，不要弄丢了。\n\n接着，我们无需额外安装依赖，只需要在项目的.env文件中添加配置即可：\n\n```Properties\n# deepseek\nDEEPSEEK_API_KEY=sk-5df6af828a04427da4d98fc53cebd63b\n# aliyun dashscope\nDASHSCOPE_API_KEY=sk-913a82aa121f412aa9a8c8c7b22f7792\nDASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n```\n\n#### 3.2 开发Agent\n\n首先，我们把刚才的代码集中到一个py文件中：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/N4OUbcx1Uo92Frxv0bjcz1rkneb/)\n\n代码如下：\n\n```Python\nfrom langchain.chat_models import init_chat_model\nfrom langchain_tavily import TavilySearch\nfrom langchain.agents import create_agent\nimport os\n\n# 1.加载环境变量\nfrom dotenv import load_dotenv\nload_dotenv()\n\n# 2.web搜索工具，使用tavily作为web搜索工具\nweb_search = TavilySearch(\n```\n\n \n> **注意**：\n>   - LangGraph会自动托管Agent的记忆，因此代码中**不用自己添加checkpointer！**\n>   - LangGraph**自带Restful的API接口**，我们只要定义好Agent就可以，其它不用管\n\n#### 3.3 本地部署\n\n我们使用LangGraph命令行在本地部署，所以要先安装LangGraph的依赖。\n\n使用uv安装：\n\n```Bash\nuv add langgraph-cli[inmem]\n```\n\n然后，在项目根目录添加一个langgraph配置文件：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BeW9bcIe8o8T4ex8cnLcSsk5nC2/)\n\n添加下面的内容：\n\n```JSON\n\n```\n\n注意：其中的`agent`配置格式为：\n\n```JSON\n\n```\n\n例如，在我们的配置中：\n\n- `./app/agents/personal_chief.py`：就是文件路径\n\n- `agent`：就是文件中定义的Agent名字\n\n最后，打开Pycharm终端：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/DWilbl7duo8pjnxZ45ccfcF3ngh/)\n\n使用LangGraph命令本地部署Agent：\n\n```Bash\n\n```\n\n效果：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/JKCLbHMjro8LFQxkVF7cwV32nhe/)\n\n看到这个，说明你的Agent已经在本地部署成功了！！\n\nLangGraph提供了基于RestFull的完整的服务端API接口，可以访问[http://127.0.0.1:2024/docs](http://127.0.0.1:2024/docs)查看。接下来，你就可以自己开发前端，与这些接口对接了。。\n\n当然，LangGraph也支持Docker部署方案，可参考以下链接：\n\n[Deploy with control plane - Docs by LangChain](https://docs.langchain.com/langsmith/deploy-with-control-plane#step-2-build-docker-image)\n\n#### 3.4 LangSmith Studio测试\n\n由于我们部署时配置了LangSmith，所以可以直接访问LangSmith提供的调试GUI界面：\n\n[https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024)\n\n这里可以非常方便的调试我们的Agent，查看我们Agent的运行细节：\n\n![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/HuBAbiz3Lo70f8xpiGDc7oHznLh/)\n\n可以直接在界面中测试：\n\n![图片](https://internal-api-drive-stream.fei\nshu.cn/space/api/box/stream/download/v2/cover/PjAObxPPbovyCgxqjScc87nwnxg/)

也可以查看详细的调用过程：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/R08Ab9F1xoYdkKxEheCckQW8nxc/)

同时，LangSmith还提供了一键云部署功能，可以把Agent部署到云端：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/WCAxboWrdofqY7xq9JycdbxgnMJ/)

但是要需要缴付昂贵的费用。。

所以，建议只在Agent测试阶段使用LangSmith吧~

### 4. 实战开发

Agent跑通了，但目前还存在几个问题：

- 目前的图片信息还是采用base64方式提交给模型，会占用大量内存，性能差

- 我们没有开发自己的前端，用户体验不好

接下来，我们就逐一解决这些问题。

在向模型提交多模态消息，比如：音频、视频、图片时，我们不建议直接发送文件数据（base64）给模型，这会大量占用内存和会话记忆。更常见的方案是：

- 先将多模态文件上传至通用的OSS服务，例如：阿里云OSS、腾讯云COS等

- 获取oss服务的文件url地址，组织多模态消息，发送给大模型

因此，我们需要单独开发一个文件上传的服务接口，让前端先上传好文件，再调用Agent.

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/NuNLbJH2fosjdbxU2wrcltR1nc1/)

我们的服务端需要具备以下接口：

- 对话接口：接收用户聊天消息，并调用Agent

- 会话管理接口：查询或删除会话历史

- 文件上传接口：调用OSS提供的客户端，实现文件上传授权，将来由前端完成文件上传，文件不经过服务器。

#### 4.1 FastAPI服务端

首先，我们需要安装一些依赖，包括FastAPI和阿里云的OSS：

```PowerShell
uv add fastapi alibabacloud-oss-v2
```

在资料中已经提供好了一个基于FastAPI开发的基础网关：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/GRVRbiBmoonF28xuY90c7Sf0nSg/)

项目结构：

```Bash

  app/
  ├── main.py                    # FastAPI 入口，配置路由和静态文件
  │
  ├── agents/
  │   └── personal_chief.py      # AI 代理核心逻辑
  │ 
  ├── api/
```

将所有内容复制到我们的app目录下：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/M9MAbjwIZoCIcnxZDuZc8uA7nqe/)

main.py作为程序入口：

```Python
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
```

直接运行main.py，访问：[http://localhost:8001](http://localhost:8001)即可：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/H2hGbczCkoe3o5xhVc4cOtNgn3E/)

#### 4.2 OSS配置

虽然资料已经帮大家实现了oss的文件上传，但是大家需要自己去oss注册服务，并申请API_KEY

这里我们以阿里云OSS为例来说明

##### 4.2.1 注册阿里云

注册地址：

[阿里云登录 - 欢迎登录阿里云，安全稳定的云计算服务平台](https://oss.console.aliyun.com/overview)

注册登录成功后，访问链接：[https://oss.console.aliyun.com/overview](https://oss.console.aliyun.com/overview)，即可看到oss控制台：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/WMPHbJmwVo2kr7xy9NScpOWOnPc/)

如果显示【**尚未开通**】，点击【**立即开通**】即可：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/QVRUbDGeIoJdxRx3pF2cyWHGnlh/)

默认为按量付费，价格非常便宜，几乎可以忽略不计：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/L9e5bYzlkomwoWx9991cSOMJnmb/)

开通成功后，即可进入控制台页面：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Ho2Ibav6co11zCxdbHhcmgH6nvf/)

##### 4.2.2 申请API_KEY

访问链接：[https://ram.console.aliyun.com/overview?activeTab=workflow](https://ram.console.aliyun.com/overview?activeTab=workflow)，进入RAM访问控制页面：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/QuZObEWNmoP0Cbx5srScdQ8Bnfb/)

点击：`用户`>`创建用户`，填写用户信息：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Wys6bJmxToOXDgxEgPmcpH6LnTd/)

创建完成后，一定要记住你的AccessKey的ID和Secret：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Pg6bbHR0UowvtlxbwAVcQr7AnWn/)

然后，给新添加的用户《新增授权》：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/K0wkb991ZoWBXixUi6QcMVU3ngd/)

在授权页面，给新用户添加oss的绝对控制权限：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/VQTTb6y6doCAlNx8em1cTzHjn7g/)

##### 4.2.3 开通OSS

访问链接：[https://oss.console.aliyun.com/overview](https://oss.console.aliyun.com/overview)，进入oss控制台，选择**创建Bucket**：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/AXmUbXgGBogpWRx7spzcFNJ2nqe/)

填写bucket信息：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Bgncb2CvaoFNUzxlw0IcBju6nyl/)

##### 4.2.4 设置权限

目前，新建的Bucket还是无法访问的，是私有的。为了方便测试，这里我们**暂时**将其设置为**公共读**。

 
> **注意**：
>   - 实际开发中oss中的图片应该设置为**私有**！不可对外暴露！！由额外的CDN服务对外暴露！
>   - 本例中，我们为了方便暂时将bucket设置为公共读，测试完毕后**请及时关闭权限**！

首先，进入Bucket管理页面：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/D5mMb7xDxoZKhDxi6A9cKaUzn3c/)

进入具体的Bucket设置，关闭公共访问开关：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/EMp7bsgcYon6fSxkayqcaTdNnNf/)

设置权限为公共读：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/VJ1Qbuo1Po4xF7xHDRTcXKmbntc/)

接着，我们要开启跨域访问权限：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/BMTAbcDMBoIO2Hx3ZqUc2vfvnpt/)

点击《**创建规则**》，填写跨域规则：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/VxbpbtdEIomng2xlA8xcmwFsnif/)

##### 4.2.5 配置API_KEY

最后，我们需要把阿里云OSS的API_KEY配置到环境变量中，也就是项目的.env文件内，格式如下：

```Properties
# 阿里OSS
OSS_ACCESS_KEY_ID=LTBI5tHzjC36KhCJfPqlbaCo
OSS_ACCESS_KEY_SECRET=aDPGBi1nIlYzcEmk5djWJGUv3w9Qkh
OSS_BUCKET=tmp9527
```

访问 [http://localhost:8001](http://localhost:8001)，试试上传图片是否成功：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Zejcbv47lovnywxiqYXcDcnWnmh/)

选择图片，可以看到预览：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/VXWgbE1fcoWcp6xjW4fcPwRVnqg/)

点击发送按钮，上传成功应该能在聊天窗看到结果：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/HEb0bFuproQo3ixYanwcutEJnJd/)

#### 4.3 添加Checkpointer

由于上一节是基于LangGraph和LangSmith的部署，所以Agent并没有添加Checkpointer，由LangGraph自己处理。而现在既然要自定义Agent部署， 就必须自己添加Checkpointer了。

我们依然使用Sqlite作为存储方案，修改personal_chief.py文件，添加Checkpointer给Agent：

```Python
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
```

在app下创建一个db目录，用以存放Sqlite的db文件：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/WuhLbHPfXolglxxKrAccSb2SnGb/)

#### 4.4 开发接口

目前在`agents/personal_cheif.py`文件中虽然已经开发了Agent，但是外界还是无法访问，我们需要定义Restful的接口供前端使用。

其中，在`api/v1/chat.py`中定义了Restful的接口，等待我们实现功能：

```Python
from fastapi import APIRouter

router = APIRouter()
```

##### 4.4.1 功能实现

我们可以在`agents/personal_cheif.py`中基于agent开发所需的3个功能：

- 多模态聊天

- 获取会话历史

- 清空会话历史

`agents/personal_cheif.py`的完整代码如下：

```Python

```

##### 4.4.2 完善接口

然后，修改chat.py文件，导入`agents.personal_cheif`中的三个方法，完善接口：

```Python
from fastapi import APIRouter
from app.models.schemas import ChatRequest
from fastapi.responses import StreamingResponse
from app.agents.personal_chief import search_recipes, get_messages, clear_messages


router = APIRouter()


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
```

#### 4.5 测试

访问[http://localhost:8001](http://localhost:8001)，即可实现多模态聊天了：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/NJdsbKTuJosyzPx8B4HcdohdnOf/)
\n