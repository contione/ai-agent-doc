# 第3章. Agent进阶

ambient

---

## 第1节. Runtime

LangChain 的 `create_agent` 底层运行在 LangGraph Runtime 之上。Runtime 可以理解为 Agent 每次执行时拿到的“运行环境”，其中与数据管理最相关的三个部分是：

- `Context`：本次调用需要的静态身份和依赖。

- `State`：执行过程中不断变化的短期状态。

- `Store`：跨会话保存的长期数据。

学完本节，应该能够：

1. 根据“是否变化”和“存活多久”区分 Context、State、Store。

1. 理解 Runtime Context 与大模型上下文窗口不是同一个概念。

1. 使用 `context_schema` 声明 Context 的数据结构。

1. 在调用 Agent 时通过 `context=` 注入运行信息。

1. 在工具内部通过 `ToolRuntime` 读取 Context。

1. 判断哪些数据适合放入 Context，哪些不适合。

### 1. 认识Runtime

Runtime是LangChain的Agent运行时对外暴露的对象，其中包含的核心数据有：

| 概念 | 作用 | 读写 | 典型生命周期 | 典型内容 | 访问方式 |
| --- | --- | --- | --- | --- | --- |
| Context | 注入本次运行所需的身份、配置和依赖 | 只读 | 单次调用 | user_id、tenant_id、权限、数据库连接 | runtime.context |
| State | 保存 Agent 执行中产生的动态数据 | 可读写 | 同一会话：同一thread_id的多次调用 | 消息、步骤结果、计数器、业务流程状态 | runtime.state |
|  |  |  |  |  |  |

判断数据放在哪里，可以连续问两个问题：

1. 这份数据在执行过程中会不会变化？不变化，优先考虑 Context。

1. 如果会变化，下一次会话还要不要使用？只在当前会话使用放 State，跨会话复用放 Store。

更多Runtime信息可以查看官方文档：[Runtime - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/runtime)

### 2. Context（运行上下文）

Runtime Context 是一种依赖注入机制：调用者在运行开始时把用户身份、租户信息、权限或外部服务连接传给 Agent，节点、中间件和工具在执行时再从 Runtime 读取。

它解决的是三个工程问题：

- 避免把用户 ID、数据库连接等信息硬编码进工具。

- 同一个 Agent 可以安全地服务不同用户和租户。

- 工具的业务参数只保留真正需要模型填写的内容。

Context 中的数据默认不会自动发给大模型；只有代码主动读取并把内容加入提示词或工具结果时，模型才会看到它。

例如，在系统中用户通常会先登录，然后访问agent，我们就可以把登录用户信息存入Context，在Agent内部的Middleware或Tool旧能很方便的通过`runtime.context`获取用户信息。而这个用户信息并不会让模型看到，也不需要作为参数让模型传递，纯粹是Agent内部信息。

#### 2.1 准备数据

下面，我们就以一个多租户的在线教育Agent系统为例，来演示Context的用法。

我们先准备一个字典，模拟一个多租户的在线教育项目数据库，其中的租户就是学校，user就是学员：

```Python
# 字典key是租户tenant_id，字典value是租户对应的user信息，也是字典。user字典的key是user_id
```

下面用“查询当前用户资料”演示 Context。模型只负责决定何时调用工具，不负责填写 `user_id` 和 `tenant_id`；工具从可信的 Runtime Context 中读取它们。

#### 2.2 定义Context Schema

首先，我们需要约定Context的数据结构，这里要使用`@dataclass`来定义，省去定义`__init__`等魔法函数

示例代码：

```Python
from dataclasses import dataclass

@dataclass(frozen=True)
class UserContext:
    """Agent运行时上下文"""
    user_id: str
```

说明：

 
> 定义UserContext类就是一个`context_schema` ，作用是给 Context 提供明确的数据结构。`frozen=True` 不是 LangChain 的硬性要求，但能表达“运行内不可修改”的设计意图。

#### 2.3 在Tool中使用Context

在tool中，我们可以添加runtime参数，然后利用`runtime.context`来访问Context：

```Python
from langchain.tools import tool, ToolRuntime

@tool
def get_current_user_profile(runtime: ToolRuntime[UserContext]):
    """查询当前登录用户的资料。"""

    # 获取context
    ctx= runtime.context

    # 根据user_id和tenant_id获取用户信息
    profile = USER_DATABASE.get(ctx.tenant_id, {}).get(ctx.user_id)
    
    return (
```

#### 2.4 给Agent添加Context

在定义Agent时，需要指定我们创建好的Context：

```Python
from langchain.agents import create_agent

# 创建Agent时指定context_schema
agent = create_agent(
```

调用Agent时，可以传递Context信息：

```Python
from langchain.messages import HumanMessage

# 调用时通过Context传递用户信息
response = agent.invoke(
    {"messages": [HumanMessage("Hello, 帮我查询我正在学习的课程")]},
```

### 3. State（短期记忆）

State我们之前有学习过，它是Agent的短期记忆，存储当前会话的**历史消息**、**任务状态**等信息。

我们之前学习短期记忆时使用的是默认的AgentState，其中只包含会话的**历史消息（messages）**，本节开始我们学习如何自定义AgentState，记录除了历史消息以外的其他信息。

#### 3.1 自定义State

要自定义AgentState，其实就是定义一个类，然后继承AgentState，在其中添加想要记录的属性信息即可。

例如，我们要实现统计用户的模型调用次数、调用时间的功能，可以这样定义state：

```Python
from langchain.agents import AgentState
```

由于在`AgentState`中已经具备`messages`属性，也就是历史消息列表，因此我们的`CustomState`继承了`AgentState`以后，不仅可以记录会话的历史消息，也能记录Agent运行的任务状态信息了。

那么，我们该如何操作state中的自定义属性呢？

LangChain中通常有两个地方可以操作state：

- `Tool`

- `Middleware`

由于Middleware还没有学习，我们先来看在Tool中访问state

#### 3.2 在工具中访问state

在定义tool的时候，LangChain内置了一个`runtime`参数，通过`runtime`我们可以获取Agent的内部信息，包括：

- `state`: dict结构

- `store`

- `context`

```Python
@tool
def my_tool(runtime: ToolRuntime):
    pass
```

 
> **注意**：
>   - runtime是了LangChain中tool的限定参数，自定义参数不能叫这个名字。
>   - `AgentState`继承自TypedDict，本质是一个`dict`，所以不能用`对象名.属性名`访问，必须用字典方式访问。

在runtime中state本质是一个`dict`，我们可以这样访问state中的属性：

```Python
model_call_count = runtime.state.get("model_call_count", 0)
```

而修改state则是通过返回一个update格式的Command指令，像这样：

```Python
@tool
def my_tool(runtime: ToolRuntime):
    return Command(
```

其中的update值是一个字典，字典结构就是自定义State的结构，其中包含要更新的字段值：

- `model_call_count`：调用计数

- `messages`：消息列表。这里必须返回一条ToolMessage，作为工具调用的结果。

例如，我们定义这样的一个tool：

```Python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage


@tool
def update_model_call_count(runtime: ToolRuntime):
    """A tool that count model call """
```

#### 3.3 设置state schema

定义了AgentState还不够，我们还需要在创建Agent时设定state schema，告诉Agent要使用自定义的state：

```Python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "deepseek-flash",
    tools=[update_model_call_count],
```

#### 3.4 测试

完整代码如下：

```Python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
```

运行结果如下：

```Python

```

通过下面的代码可以查看state信息：

```Python
agent.get_state(config)
```

结果如下：

```Plain Text

```

### 4. Store（长期记忆）

store是LangChain提供的长期记忆机制，用于在不同会话间共享数据。例如：模型以外的数据、用户偏好等。

例如，同一位用户可能先在 `thread-a` 中告诉 Agent“我喜欢代码示例优先”，几天后又在 `thread-b` 中开始新的课程咨询。

如果这条偏好只放在 State 中，新 thread 无法自动读取；把它写入 Store 后，两个 thread 都可以访问。

LangChain提供了多种Store的实现方式，例如：

- InMemoryStore

- PostgresStore

- RedisStore

- ...

课程中我们以InMemoryStore为例来学习。

#### 4.1 Store的数据结构

Store的数据格式是JSON文档，JSON文档采用分级管理：

- Namespace（命名空间）：可以理解为一个文件夹

所以，定位Store中的一条数据，需要两部分信息：`namespace + key`，其中`namespace`是一个元组，而`key`则是普通字符串。

例如，在多租户的在线教育Agent，我们要记录每个用户的学习偏好，namespace和key分别可以采用下面的组织方式。

| namespace | key | value |
| --- | --- | --- |
| 元组结构 Python | "learning_preferences" |  |
|  |  |  |
|  |  |  |
|  |  |  |

由于Store的结构是固定的`namespace + key -> json文档`的格式，所以无需我们定义结构，我们只要确定好自己要用的namespace和key，就可以直接使用。

```Python
namespace = (
    "heima_platform",  # 黑马程序员平台
    "school_a",        # 租户id，例如某个校区
    "u002"             # 用户id，也就是学员
)
```

#### 4.2 初始化Store

首先，我们要初始化Store：

```Python
from langgraph.store.memory import InMemoryStore

# ==================== 1. 创建Store ====================
store = InMemoryStore()
```

store中有两个方法，分别用来读取和写入Store：

- `store.get(namespace, key)`：读取Store

- `store.put(namespace, key, value)`：存入Store，这里的value必须是`dict[str,Any]`结构，而且dict中的值必须是可以转JSON的，LangChain会将其转JSON存储

#### 4.3 在tool中访问store

与state类似，LangChain中访问store通常也可以有两种场景：

- Tool

- 中间件

我们依然先学习tool中访问store，与state类似，在tool 中访问store也是通过runtime

在tool中访问store要这样做：

```Python
from langchain.tools import ToolRuntime, tool


# 定义方法，根据Context中的tenant_id和user_id拼接得到namespace
def preference_namespace(ctx: UserContext) -> tuple[str, ...]:
    return (
        "heima_platform",
        ctx.tenant_id,
```

#### 4.4 给agent添加store

创建agent时需要指定store：

```Python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


checkpointer = InMemorySaver()
store = InMemoryStore()

agent = create_agent(
    model="deepseek-flash",
    tools=[
        get_current_user_profile,
```

测试：

```Python
context = UserContext(
    user_id="u001",
    tenant_id="school_a",
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "记住：我喜欢先看代码，再看原理",
            }
        ]
```

运行结果：

```Plain Text
================================ Human Message =================================

记住：我喜欢先看代码，再看原理
================================== Ai Message ==================================
```

```SQL

```

输出结果：

```Markdown

```

### 5. 总结

1. **Context（运行时上下文）**

1. **State（短期记忆）**

1. **Store（长期记忆）**

---

**更多资源**:

- [LangChain官方文档 - Runtime](https://docs.langchain.com/oss/python/langchain/runtime)

- [LangChain官方文档 - Long-term Memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)

- [LangGraph Store文档](https://langchain-ai.github.io/langgraph/concepts/store/)

---

## 第2节. Middleware

**Middleware**，也就是**中间件**，是一种控制Agent内部运行过程的技术，它在智能体运行的各个过程中预留钩子（hook），方便我们嵌入自定义操作。

Agent的默认执行流程和中间件嵌入钩子的对比如图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/SevXbTGBxo1E1jxdOe0cce15n9b/)

**Agent核心执行循环**

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/PopUbJldVoEAnBxz5sYcnzHgnuf/)

**Agent核心执行循环**

基于Middleware可以实现各种高级功能，比如：

- **拦截和修改请求** - 在模型调用前后对输入输出进行处理

- **实现PII脱敏** - 自动检测和脱敏敏感个人信息

- **会话摘要管理** - 当对话过长时自动压缩历史消息

- **人工审核机制** - 在执行危险操作前等待人工确认

- **动态模型选择** - 根据运行时条件选择不同的模型

- **自定义状态管理** - 扩展Agent状态以跟踪额外信息

Middleware可以自己定义，也可以使用LangChain预定义好的。

### 1. 预定义中间件（Prebuilt Middleware）

LangChain提供了多种预置中间件，可以直接使用。完整内置Middleware列表可以参考官方文档：

[Prebuilt middleware - Docs by LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

我们会以其中的3个为例来介绍预定义中间件的用法：

- PIIMiddleware

- ModelFallbackMiddleware

- HumanInTheLoopMiddleware

#### 1.1 PIIMiddleware - 个人信息脱敏

PIIMiddleware是一种预定义的wrap_model_call中间件，可以在调用模型前后自动检测并脱敏输入、输出消息中的个人身份信息（PII），如邮箱、电话号码、身份证号等。

其中，PII的脱敏处理策略有四种：

- `'block'` - 抛出异常

- `'redact'` - 用 [REDACTED_{PII_TYPE}] 来替代

- `'mask'` - 关键信息采用**掩码 (例如., `****-****-****-1234`)

- `'hash'` - 用哈希值来替换

示例：

```Python

```

由于手机号的处理策略是block，所以运行时会抛出异常：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/ZRsebqjbvoRDYWxQZu7cQPkrnzg/)

#### 1.2 ModelFallbackMiddleware

ModelFallbackMiddleware的作用是在模型调用失败时给出降级处理方案。可以在创建时设置多个模型，如果主模型调用失败，会自动调用备用模型。

示例：

```Python
from langchain.agents.middleware import ModelFallbackMiddleware

model_fallback_middleware = ModelFallbackMiddleware(
    "deepseek-v",  # 默认模型
    "deepseek-flash"  # 备用模型
)
```

由于deepseek-v模型不存在，所以调用时会走备用模型deepseek-chat

```Python

```

#### 1.3 HumanInTheLoopMiddleware - 人工审核

HumanInTheLoop，简称为**HITL**，其作用是让人工介入到Agent执行流程中，在执行工具调用前暂停，等待人工确认。

通常用于Agent执行敏感操作前的确认，例如：

- 发送邮件

- 转账

- 执行脚本

- 读写文件

- ...

##### 1.3.1 创建带有HITL的Agent

假如我们有一个负责邮箱操作的智能体，示例代码如下：

准备假的邮件数据：

```Python
emails = [
        {
            "subject": "周末见个面？",
            "content": """
```

**首先**，定义查看邮件、发送邮件的tool：

```Python
from langchain.tools import tool

# 定义工具，收取和发送邮件
@tool
def check_inbox() -> str:
```

**然后**，定义Middleware，设置需要人工确认的tool，以及人工确认的可选操作：

```Python
from langchain.agents.middleware import HumanInTheLoopMiddleware
```

代码解读：

- `interrupt_on`：就是需要人工确认的tool信息，可以设置多个

- *`send_email`*：就是需要人工确认的tool名字

- `description`：是人工确认时的提示信息

- `allowed_decisions`：是人工确认时的可选操作，包括三种（**可选**）：

在上例中，我们对`send_email`做人工确认，但`check_inbox`并没有。

**最后**，创建智能体，设置Middleware：

```Python

```

接下来，测试一下：

```Python

```

由于check_inbox没有做人工确认，所以会直接执行：

```SQL
================================ Human Message =================================

帮我查看邮件
```

我们让Agent回复邮件：

```Python
response = agent_with_hitl.invoke(
```

可以看到，AI确实去调用了`send_email`，但工具并没有执行，没有看到Tool Message回复：

```Plain Text
============================== Human Message =================================

帮我查看邮件
```

如果打印response，可以看到工具调用被中断了（Interrupt）：

```Plain Text

```

```Python

```

此时，如果Agent对接了前端，就应该在页面展示**需要人工确认的提示信息**：'请确认邮件内容'，并给出`approve`、`reject`、`edit`这3个选项。

当用户选择一个操作后，将用户的选择告诉Agent。

那么问题来了，用户选择操作后，该如何将用户的选择告诉Agent呢？

##### 1.3.2 reject

如果用户拒绝Agent调用工具，我们需要再次调用Agent，并通过Command来告知Agent用户选择了reject，并且告知拒绝原因。

示例：

```Python
from langgraph.types import Command

response = agent_with_hitl.invoke(
    Command(
        resume={
            "decisions": [
                {
```

结果，可以看到AI重新生成了邮件内容，并再次调用`send_email`，并再次触发人工确认：

```Plain Text

```

##### 1.3.3 approve

这次，我们调用`approve`，表示同意调用：

```Python
from langgraph.types import Command

response = agent_with_hitl.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config  # 相同的thread_id以恢复暂停的对话
)

for message in response['messages']:
    message.pretty_print()
```

结果：

```YAML
================================ Human Message =================================

帮我查看邮件
================================== Ai Message ==================================

我来帮你查看收件箱。
Tool Calls:
  check_inbox (call_00_KmKmhdkUjmbEQtrK6YSn5875)
 Call ID: call_00_KmKmhdkUjmbEQtrK6YSn5875
  Args:
================================= Tool Message =================================
Name: check_inbox
```

### 2. 自定义中间件

除了使用预置中间件，我们还可以利用LangChain提供的hook创建完全自定义的中间件。

根据hook的种类，中间件可以分为两类：

- Node-style hooks：在具体某个节点执行的中间件，包含：

- Wrap-style hooks：环绕model或tool调用的中间件，包括：

为了便于开发中间件，LangChain为每一种hook都提供了装饰器，我们只要定义函数并使用装饰器标记即可快速开发中间件。

#### 2.1 node-style装饰器

- Node-style hooks：在具体某个节点执行的中间件，包含：

例如，我们要实现模型调用计数功能，之前利用tool来统计增加了很多次tool调用，浪费token，而且也不太方便。利用中间件就优雅很多。我们可以在每次调用模型后(**after_model**)记录模型调用次数。

做法是定义函数，并用`@after_model`装饰器来装饰该函数，但要注意，**函数的参数和返回值必须严格按照下面的示例**：

```Python
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from langchain.agents.middleware import after_model


# 1.定义自定义state，记录模型调用次数
class CustomState(AgentState):
    """扩展Agent状态，添加自定义字段"""
    model_call_count: int  # 模型调用次数


# 2.定义中间件
@after_model(state_schema=CustomState)
def increment_counter(
```

测试：

```Python
# 创建智能体，设置middleware
agent = create_agent(
    model="deepseek-chat",
```

结果：

```Python
{'messages': [HumanMessage(content='Hello', additional_kwargs={}, response_metadata={}, id='f333c2ff-d50b-452c-941b-adb1c69812f6'), AIMessage(content='你好！有什么可以帮你的吗？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 10, 'prompt_tokens': 5, 'total_tokens': 15, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 5}, 'model_provider': 'deepseek', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'fp_058df29938_prod0820_fp8_kvcache_20260402', 'id': 'fc0a7204-aafb-44f2-89a0-c516096134b1', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019dbe9e-f818-7b82-86ab-dd39577897b2-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 5, 'output_tokens': 10, 'total_tokens': 15, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})], 'model_call_count': 1}
```

#### 2.2 wrap-style装饰器

- Wrap-style hooks：环绕model或tool调用的中间件，包括：

例如，我们来定义一个可以在调用模型时失败重试的中间件，最大重试次数为3次。那就可以使用`@wrap_model_call`，在模型调用时做出判断，如果失败则重试，重试此时超过3次则结束。

同样的，被`@wrap_model_call`装饰的函数，其参数和返回值必须严格按照下面的格式：

```Python
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from typing import Any, Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
```

测试：

```Python
config = {"configurable": {"thread_id": "1"}}

# 调用智能体，并初始化state
response = agent.invoke(
    {"messages": [HumanMessage("Hello")]},
    config,
)

print(response)
```

#### 2.3 类装饰器

对于需要同时用到多个hooks的更复杂的中间件逻辑，我们还可以使用自定义类继承AgentMiddleware的方式来创建中间件。

例如，一个用来记录日志的中间件，要在模型调用、工具调用前后记录日志：

```Python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ToolCallRequest
from langgraph.types import Command


class LoggingMiddleware(AgentMiddleware):
```

运行结果：

```SQL

=======About to call model with 1 messages=======
好的，我来查询一下杭州今天的天气情况。
=======调用工具: get_weather=======

=======参数: {'location': '杭州'}=======

=======工具调用成功！=======
Current weather in 杭州 is sunny, 25℃.
=======About to call model with 3 messages=======
杭州今天天气晴朗，气温 **25℃**，非常适合外出活动哦！ 注意适当防晒～
```

### 3. 高级用法

中间件除了利用hook做基本的信息记录和判断，还可以有一些高级的用法，例如：

- 动态修改请求：可以拦截发送给模型的请求，动态修改请求中使用的模型、工具、提示词等

- 条件跳转：在满足条件的情况下直接跳转到某个Agent执行的节点

#### 3.1 动态修改请求参数

在wrap_model_call这个hook中，我们可以动态修改任意的request参数，包括：

- model

- tool

- system_prompt

- ...

关键点是利用`request.``override()`来覆盖原本的request参数。

例如，我们定义中间件，可以根据context中用户的选择来开启或关闭模型的思考模式。

```Python
from langchain.agents.middleware import wrap_model_call
```

测试：

```Python

```

#### 3.2 条件跳转

在中间件使用`jump_to`指令可以跳过模型调用，直接进入指定的Agent节点。

例如，我们设定一个模型限流的中间件，当判断用户模型调用次数超过阈值，直接禁止调用，结束Agent运行。

```Python
from langgraph.runtime import Runtime
from typing import NotRequired, Any
from langchain.agents.middleware import AgentMiddleware, hook_config, AgentState
```

测试：

```Python
# 创建智能体，设置middleware
agent = create_agent(
    model="deepseek-chat",
    middleware=[ModelCallLimitMiddleware(max_limit=2)],
    checkpointer=InMemorySaver(),
    state_schema=CustomAgentState
)
```

---

## 第3节. MCP

**MCP **(**M**odel **C**ontext **P**rotocol) 是由 Anthropic 推出的开放标准，用于便捷的将AI应用连接外部系统。

在没有MCP的时候，必须手动定义工具，用工具实现文件操作、web搜索、查询航班、查询天气等功能，从而让AI连接外部系统。

这就存在两个问题：

- 不同Agent可能有同样的tool需求，每次都重复定义，复用性差

- 全世界有各种不同的服务，不同服务接口不同，定义Tool非常麻烦

MCP就像是AI世界的USB接口协议：

- 所有外部服务提供者都可以遵循MCP协议提供Tool，分享自己的Tool服务

- AI应用基于MCP协议对接任意遵循MCP的外部服务，无需自己定义Tool

这样一来就解决了重复定义工具的复用性问题、以及对接全世界各种公共服务的问题。

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/G8YZbYTJboZadix42PAc5WzTnSb/)

### 1. MCP核心概念

在MCP中有三个核心概念：

| 概念 | 说明 |
| --- | --- |
| MCP Server | 提供MCP服务的应用，可以是远程服务，也可以是本地服务 |
| MCP Client | 连接到MCP服务器，读取MCP信息（特别是Tool信息），供Host使用 |
| MCP Host | 协调和管理多个MCP Client的AI应用，比如LangChain的Agent |

例如，一个AI应用，也就是MCP Host，它需要三个功能：

- 文件操作

- 数据库操作

- Sentry远程服务

此时，它可以定义3个不同的MCP Client，分别对接3个MCP Server，包含一个操作本地文件的MCP、一个访问数据库的MCP、一个访问Sentry服务的MCP

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/SxFcbWqReoWe4dxby6zcimRsn9g/)

MCP Client 与 MCP Server之间有两种通信协议：

- stdio

- streamable_http

stdio就是标准输入输出，MCP Client运行时，分两种情况：

- 外部服务：Client会把这个MCP服务的脚本下载到本地，然后作为一个子进程运行。

- 本地服务：Client会把本地脚本直接加载，作为一个子进程运行

也就是说，stdio模式中，MCP Client 和 MCP Server之间的通信就是进程通信，没有网络延迟。

streamable_http其实就是可以用event stream来发送数据的http模式，本质还是Server Send Event，也就是SSE。也就是说MCP client通过发送http请求与MCP server交互。因此存在一定的网络延迟。

详细的MCP说明可以参考Anthropic公司的MCP官方文档：

[https://modelcontextprotocol.io/docs/getting-started/intro](https://modelcontextprotocol.io/docs/getting-started/intro)

有关通信协议可以查看：

[https://modelcontextprotocol.io/specification/2025-11-25/basic/transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

LangChain本身实现了对MCP的支持，本节我们就来学习如何在LangChain中使用MCP

### 2. 连接外部MCP服务

很多提供云服务的公司都提供了MCP服务，例如：

- Amap Maps : 高德地图提供的MCP

- Filesystem : 可以操作文件系统的MCP

- Time : 查询当前时间的MCP服务

- Kiwi : 查询航班、预定航班的MCP服务

- ...

大家可以在[https://mcp.so/zh/](https://mcp.so/zh/)搜索各种MCP服务（国际的比较多）：

[MCP.so](https://mcp.so/zh/)

也可以在魔塔社区搜索MCP服务（国产的比较多）：

[ModelScope - MCP 广场](https://modelscope.cn/mcp)

找好自己想要使用的MCP服务后，就可以用LangChain来对接了。

首先，我们需要安装LangChain的MCP client依赖库:

```Plain Text
uv add langchain-mcp-adapters
```

接下来，就可以用LangChain对接MCP服务，获取其提供的工具，创建Agent了。

接下来，我们以两个MCP服务为例来介绍LangChain对接MCP服务的方式：

- Time MCP ：基于stdio通信

- Kiwi MCP : 基于Streamable http通信

#### 2.1 Time MCP服务

Time是一个提供时间和时区转换功能的MCP服务。此服务使LLM能够获取当前时间信息，并使用 IANA 时区名称执行时区转换，同时具备自动检测系统时区的功能。

查看Time MCP的介绍页面，可以看到其介绍，以及集成方式：

[MCP Server（MCP 服务器）](https://mcp.so/zh/server/time/modelcontextprotocol)

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/U6H0bER9doEfHzxUwzscCqkQnec/)

需要注意的是，Time MCP是基于**stdio**的通信方式，也就是把这个MCP服务的脚本下载到本地，然后作为一个子进程运行。

其中，MCP脚本下载的方式取决于配置中的`command`，常见的有两种：

- `npx` : 基于node.js的包管理工具

- `uvx` : 基于uv(python的uv工具)的包管理工具

因此，你的本地环境必须支持npx、uvx命令

另外，在LangChain中除了基本的服务器配置，还必须设置一个`transport`参数，指定MCP服务的通信方式，有两个可选值：

- `stdio`

- `http`或者`streamable_http`

与标准的mcp参数相比，LangChain的配置存在一些差异：

```JSON
{
  "mcpServers": {
    "time": {
```

```JSON
client = MultiServerMCPClient(
    {
        "time": {
```

**注意**：如果是在**Notebook**运行，可能出现事件循环错误，需要加一段判断逻辑：

```Python

```

示例代码（**以下代码必须在Notebook中运行**）：

```Python
# 1.加载环境变量
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

# 2.初始化模型
model = init_chat_model(
```

结果：

```JSON
================================ Human Message =================================

现在几点了？
================================== Ai Message ==================================

好的，我先获取一下当前的时间。
Tool Calls:
  get_current_time (call_00_LEeOqXUsKTiAo4abwEjn6951)
 Call ID: call_00_LEeOqXUsKTiAo4abwEjn6951
```

#### 2.2 Kiwi MCP服务

Kiwi Travel MCP 将 [Kiwi.com](http://Kiwi.com) 的航班搜索功能直接引入到 AI 对话中。支持单程/往返、灵活日期、多名乘客以及所有舱位等级。

 
> **注意**：
>   - Kiwi服务是基于http的通信方式，也就是说MCP client通过发送http请求与MCP server交互，在国内访问存在较大网络延迟。
>   - 通常这类外部的MCP服务都是需要收费的，需要申请API key才能使用，而Kiwi是仅有的免费航班搜索服务

查看Kiwi介绍页，可以找到它的使用方式：

[MCP Server（MCP 服务器）](https://mcp.so/zh/server/kiwi-travel-mcp/Vytautas%20Dargis)

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/FMNubsPcyocImExZGJnctnkMnRg/)

示例代码：

```Python

```

 
> **注意**：由于Kiwi中有地域参数（locale），但Tool描述中没说清楚，中文有两种：zh-cn,zh-tw，需要在系统提示词中说明，否则AI默认会传入zh，导致报错。

结果：

```JSON

```

 
> **说明**：Kiwi更擅长搜索国际航班，如果是国内航班更建议大家基于携程官方API开发Tool来实现航班搜索。

另外，有一个替代的国内航班查询MCP服务，地址是：[https://mcp.variflight.com/](https://mcp.variflight.com/)

不过这个服务是收费的，而且价格比较贵，即便是优惠过后也需要0.1元/次。

使用步骤是：

- 注册账号

- 申请API Key

- 接入MCP

代码示例：

```Python

```

### 3. 自定义MCP服务

在公司内部，不同团队之间也可以把自己团队的服务开发成MCP Server，供其它团队使用。

接下来，我们就学习如何自定义MCP服务。

#### 3.1 创建简单的MCP服务器

自定义MCP最简单的方式就是使用FastMCP了。

首先，需要安装依赖：

```Plain Text
uv add fastmcp
```

接着，只需要定义几个方法，然后利用FastMCP提供的装饰器即可：

- `@mcp.tool` : 作为MCP中的工具

- `@mcp.resources` : 返回MCP需要的resources，类似拓展知识库

- `@mcp.prompt `: 返回MCP预定义的Prompt，预设的提示词

 
> **说明**：MCP server端不仅可以提供tool，还可以提供resource、prompt，但通常不太常用，一般只需要提供tool即可。

通常我们只需要定义带有tool的MCP Server就可以了。

例如，我们定义一个数学运算的MCP服务，这个需要写到单独的py文件中，比如`math_mcp_server.py`：

```Python
from fastmcp import FastMCP

# 初始化mcp
mcp = FastMCP("Math")
```

一个自定义MCP server就准备好了。

注意，由于我们是采用stdio方式，因此这个文件写好放在那里，不需要启动，将来MCP Client会自己启动并加载为子进程。

#### 3.2 连接接自定义MCP服务

由于我们自定义的MCP是本地py文件，所以启动的Command直接就是python，而不是npx或uvx

```Python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

# 连接自定义MCP服务
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_mcp_server.py"] # 这里是脚本所在的路径，此处是相对路径
        }
```

结果：

```Python
================================ Human Message =================================

467和529的平方根之和是多少?
================================== Ai Message ==================================

我们先分别计算467和529的平方根，再求和。

首先计算529的平方根：
```

### 4. 总结

1. **MCP是什么**

1. **连接方式**

1. **使用流程**

1. **自定义MCP Server**

---

## 第4节. Multi Agent

Multi Agent，既多智能体协作系统，专门用来处理复杂的任务流程。

### 1. 概述

大多数情况下，使用合适的Tool、合适的提示词和模型，单智能体就能完成任务。我们也推荐大家优先使用单智能体，只有在一些特殊情况下才推荐使用多智能体系统。

#### 1.1 使用场景

通常在以下几种情况下我们会使用Multi Agent：

- **上下文管理(Context Management)**：如果同时需要调用的工具很多，或者上下文内容很多，我们可以将任务拆分，交给不同的Agent处理

- **分布式开发(Distributed development)**：不同的团队独立开发和维护自己的Agent，并将他们组合成一个更大的Agent

- **并行(Parallelization)**：将任务拆分为多个子任务，并交给专门的Agent处理，并同时执行它们以加快处理速度

#### 1.2 常见模式

多智能体协作的模式有很多种，比较常见的有：

- **Subagents**：**子代理模式**，一个主Agent将多个子Agent作为Tool来协调使用。所有请求都由主Agent处理，决定何时以及如何调用每个子Agent

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/HBdXb82U2oiCbPxTxdJc9a1SnuQ/)

- **Handoffs**：**传递模型**，随着任务的执行改变state中的任务状态，从而触发路由变更或者触发Agent的配置变更，从而切换到其它Agent或者改变Agent的工具或系统提示（类似与一个新agent）。因此每个Agent都可以与用户交互，处理用户请求并返回响应。

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Wu76bXL4jo4Lv7xyP10cgbOyn1e/)

- **Skills**：**技能模式**，只有1个Agent，根据任务按需加载Skill或知识

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/LFHUb0XTZoY2wpxMacdcJCYlnSf/)

- **Router**：**路由模式**，1个负责路由的Agent对用户请求进行分类，将请求导向给一个或多个专门的Agent。最后再由一个Agent负责总结结果。

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/KmJKbF5sRojaxRxd5micnadznMd/)

我们从四个方面来对比这几种模式：

- Distributed development：是否支持不同团队独立开发维护

- Parallelization：是否支持并行运行多个subagent

- Multi-hop：是否支持按照特定顺序依次执行多个subagent

- Direct user interaction：是否支持subagent直接与用户对话

对比如下：

| Pattern 模式 | Distributed development | Parallelization | Multi-hop | Direct user interaction |
| --- | --- | --- | --- | --- |
| Subagents | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| Handoffs | - | - | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Skills | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Router | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | - | ⭐⭐⭐ |

 
> 当然，在实际开发中不局限与这四种模式，事实上我们可以把这些模式任意的混合使用，创造出无限的可能性。

### 2. 案例

接下来，我们就做一个多Agent案例。

现在，我们计划开发一个婚礼策划智能体，它包含以下三个核心功能：

- 旅行规划：负责为你和宾客前往婚礼目的地寻找合适的机票，制定旅行计划

- 场地规划：负责根据宾客人数在目的地寻找合适的婚礼场地

- 音乐规划：负责根据用户需求筛选合适的婚礼歌单，并计算出预算

#### 2.1 需求分析

理论上说，我们只要准备好以上每一步所需的**工具**，写好包含完整流程的**系统提示词**，用一个Agent就能实现这个功能。

但是，考虑到以下几个原因：

- 要使用的工具实在是太多，航班、场地、音乐等要查询的信息非常多，这会导致单Agent的上下文非常大，有可能超出模型上下文限制。把任务拆解可以减少上下文需求，因此适合分布式开发。

- 旅行、场地、音乐的三个任务没有关联，为了提高效率，可以并行执行

综上，建议采用多Agent开发，模式可以选择Subagents模式。我们可以开发三个Subagent：

- travel agent：负责为你前往婚礼目的地寻找往返机票

- venue agent：负责根据宾客人数在网上搜索合适的婚礼场地

- playlist agent：负责在音乐数据库筛选符合用户需求的歌单，并计算出预算

最后，我们还会定义一个主Agent，负责协调（*Coordinator*）工作，以及生成最终的婚礼计划方案。

当然，我们还是在jupyter中开发测试。

首先，需要解决mcp在Windows平台运行的问题：

```Python
import sys

if sys.platform == "win32":
    if "ipykernel" in sys.modules:
        sys.stderr = sys.__stderr__
```

然后是完整的依赖：

```Python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentState
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.types import Command
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()
```

接下来，就是agent开发了。

#### 2.2 travel agent

旅行agent要查询机票信息，需要Time和Kiwi两个MCP服务。

我们先定义工具：

```Python
# MCP客户端，包含Time、Kiwi两个MCP
client = MultiServerMCPClient(
```

然后是agent：

```Python
# Travel agent
travel_agent = create_agent(
    model="deepseek-flash",
    tools=tools,
    system_prompt="""
    你是一个旅行助手Agent。你为用户搜索到达婚礼地点的理想航班，您不能再追问任何后续问题。
    您必须根据以下标准找到最佳航班选择：
    - 价格（最低，经济舱）
    - 持续时间（最短）
    - 日期（您认为在该地点举行婚礼的最佳时间）
```

#### 2.3 venue agent

接着是负责婚礼场地的agent，这里我们简化实现方案，直接基于tavily搜索场地。

因此，首先是定义Tavily的web_search工具：

```Python
# 定义Tavily web_search工具
web_search = TavilySearch(
    max_results=5,
    topic="general",
)
```

然后是agent：

```Python
# 创建 Venue agent
venue_agent = create_agent(
    model="deepseek-flash",
    tools=[web_search],
```

#### 2.4 playlist agent

最后，是负责婚礼歌单的agent，为了方便查询，我提供好了一些模拟数据，保存在Sqlite中。db文件在notebooks的resources目录可以找到：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/PdccbGuvGoF1qhxTnZDcHei3nod/)

我们直接定义一个用于执行SQL语句的Tool：

```Python
db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:

    """Query the database for playlist information"""

    try:
        return db.run(query)
    except Exception as e:
```

测试一下：

```Python
# 测试，查询歌单数据
query_playlist_db.invoke({"query":"SELECT * FROM Playlist"})
```

如果能查询出下面数据，说明没问题：

```Python
"[(1, 'Music'), (2, 'Movies'), (3, 'TV Shows'), (4, 'Audiobooks'), (5, '90’s Music'), (6, 'Audiobooks'), (7, 'Movies'), (8, 'Music'), (9, 'Music Videos'), (10, 'TV Shows'), (11, 'Brazilian Music'), (12, 'Classical'), (13, 'Classical 101 - Deep Cuts'), (14, 'Classical 101 - Next Steps'), (15, 'Classical 101 - The Basics'), (16, 'Grunge'), (17, 'Heavy Metal Classic'), (18, 'On-The-Go 1')]"
```

接下来就是agent：

```Python
# Playlist agent
playlist_agent = create_agent(
    model="deepseek-flash",
    tools=[query_playlist_db],
    system_prompt="""
    你是播放列表专家。查询sql数据库并为给定类型的婚礼策划完美的播放列表。
    一旦你有了你的播放列表，计算播放列表的总时长和成本，每首歌都有一个相关的价格。
```

#### 2.5 主Agent

最后，就是主Agent了，也是整个Agent的核心协调者。它负责与用户交互，收集婚礼信息，调用各个Subagent，筹备婚礼计划。

##### 2.5.1 定义state

首先，我们需要定义一个state，记录婚礼风格有关的信息，包括：

- 婚礼人数

- 音乐风格

- 出发地

- 婚礼举办地

```Python
class WeddingState(AgentState):
    origin: str         # 起点
    destination: str    # 目的地
    guest_count: str    # 宾客数量
```

##### 2.5.2 Tools

首先，要执行婚礼规划任务必须知道婚礼信息，也就是state中要记录的数据。AI通过与用户沟通可以获得这些信息，但是必须有一个tool负责更新state信息，当AI通过与用户聊天获取这些信息后，调用该tool更新到state中。

```Python
@tool
def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> str:
    """当你询问用户，知道下列所有值时更新State：origin、destination、guest_count、genre"""
```

接下来，按照Subagent模式，主Agent需要把Subagents当做一个个Tool来协调使用。所以，我们接下来就先定义3个Tool，分别对应3个Subagent.

这些tool读取state中的婚礼信息，然后调用对应的subagent，分别完成自己的任务。

```Python
@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """旅行Agent搜索飞往理想婚礼地点的航班."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]

    content = f"查询从{origin}到{destination}的航班"
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content)]})
    return response['messages'][-1].content

@tool
def search_venues(runtime: ToolRuntime) -> str:
    """场地Agent根据给定的地点和宾客数量寻找最佳的婚礼场地."""
    destination = runtime.state["destination"]
```

##### 2.5.3 Agent

接下来，就是创建主Agent了，把刚刚的4个Tools都注册给它，并给它设定核心协调者的任务。

```Python
from langchain.agents import create_agent

coordinator = create_agent(
    model="deepseek-flash",
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="""
    你是婚礼协调员。你负责与用户沟通，找到更新状态所需的所有信息，并更新到state。
    一旦完成，你就可以委派任务了。将查找航班、搜索场地和策划播放列表的任务委派给您的专家Agent。
    当你收到他们的答案，为我设计一场完美的婚礼方案。
```

测试：

```Python
response = await coordinator.ainvoke(
    {
        "messages": [HumanMessage(content="我来自北京，我想在上海举办一场100人的婚礼，爵士风格的。")],
    }
)
```

运行结果如下：

```Markdown
================================ Human Message =================================

我来自北京，我想在上海举办一场100人的婚礼，爵士风格的
================================== Ai Message ==================================

好的！我收到了您的信息：
- 出发地：北京
```
