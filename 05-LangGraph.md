# 第5章. LangGraph

LangChain已经能帮助我们开发智能体、RAG系统了，为什么还要学习LangGraph呢？

要明白这个问题，我们需要先弄清楚LangGraph到底能做什么。

让我们从AI应用的发展过程谈起。

# 1. AI应用的发展

## 1.1 简单LLM应用

在早期，AI应用就是简单的调用LLM：

> [白板/画板内容] (token: X7zudfd0RoV41uxM9rGc02BdnJJ)

这时候的AI应用能力有限：

- 无法调用工具

- 无法访问外部数据

- 无法实现复杂多步骤任务

## 1.2 链式LLM应用（Chain）

为了增强LLM应用的能力，人们开始自定义程序执行流程，添加额外的步骤：

> [白板/画板内容] (token: O0sodH3BzosiKwxUG7gcYwBSnrd)

例如，在调用LLM前加入**知识检索**（retrieval），在模型调用之后加入**工具调用**（tool calls），此时整个由程序控制的工作流就形成了一个**链条**（**Chain**）。早期的LangChain也就是基于这种理念设计，这也是LangChain名字的由来。

这种人工定制的工作流特点是**稳定可靠（Reliable）**，你不管执行多少次，流程一定是这样，不会出现意外情况。但是它缺乏**自主智能（Agentic）**，距离真正的智能体相差甚远**。**

## 1.3 智能体（Agent）

所谓的**智能体**，就是指应用的执行工作流应该由LLM来决定，也就是让AI来自主规划工作流。

例如，由LLM来决定接下来应该执行Step 2 还是 Step 3：

> [白板/画板内容] (token: G3nbdSqSJoaX5JxIVeYc3iCenhe)

甚至，完全由LLM自主决定所有的执行流程：

> [白板/画板内容] (token: HUqEdhdhOoL273xrIgTc0cG0nV3)

## 1.4 稳定度与自主控制力

虽然都是Agent，但上面的两个例子恰好是Agent的两个极端，或者说是LLM对应用控制能力的两个极端：

> [白板/画板内容] (token: QvpKdd7p3omYF3xebvNcezj6nSg)

但是，Agent的自主控制能力与稳定程度之间是对立的两端，通常来说，LLM自主控制的越多，应用的稳定程度就会越小：

> [白板/画板内容] (token: H2ygds9RUouY7bxB2HtcXGtWnKf)

所以，通常来说，我们只能在可靠度和自主控制程度之间选择一个平衡点。

# 2. LangGraph的作用

LangGraph的作用就是在保持Agent的自主控制程度不变的同时，提高Agent的稳定程度：

> [白板/画板内容] (token: BoPkdVy7SoHudPx1ebIcIofMnde)

那么，LangGraph是如何做到的呢？

## 2.1 认识LangGraph

简单来说，LangGraph允许你自己定义Agent的**工作流**（WorkFlow）中的每一个**节点**（Node）：

- 需要稳定性的节点，就用传统编程控制

- 需要LLM自主控制的，就交给LLM

并且，整个工作流不再是一条直线运行的**链（Chain）**，而是一个像真正工作流那样有分支、有循环的**图（Graph）。**

在LangGraph中，工作流由3个基本的要素组成：

- **节点（Node）**：即工作流中的关键工作代码，可以是工具调用、知识检索、LLM调用

- **边（Edge）**：也可以理解为线，就是把各个节点连接起来的路径，是控制工作流走向的关键

- **状态（State）**：整个工作流中流转的数据

> [白板/画板内容] (token: OZgJdv3tfoD44ixlTENcJUAdnYa)

为了提高应用的运行的稳定程度，LangGraph为节点运行提供了Checkpointer功能，应用运行到每个**节点（Node）**都会形成**检查点（Checkpoint）**，我们可以方便的跳转到任意checkpoint，这样一来应用就具备了三大功能：

- **故障恢复**：即便是程序异常中止，也可以随时恢复到失败节点继续执行。

- **人机交互**：在程序运行中可以暂停工作流，在关键节点允许人工介入，得到人工确认后恢复执行，进一步提高程序的可靠性。

- **持久记忆**：由于Checkpointer可以持久化存储，能方便的拿到历史信息，因此应用还具备了记忆功能。

## 2.2 LangGraph与LangChain

看完上面的介绍，你是不是觉得这些功能似曾相识啊？

没错，在我们之前学习的LangChain开发中就有这些能力：

- Memory

- HITL

- ...

这是因为，自从LangChain的1.0版本以后，底层的Chain模式已经逐渐废弃，而是彻底的投入了Graph的怀抱。

LangChain 1.x之后的版本中，Agent的底层实现已经全部替换为基于LangGraph了！

不过，两者的定位完全不同：

- **LangChain**：便捷开发AI应用的框架，提供了开发AI应用的各种统一API

- **LangGraph**: 工作流编排框架，提供了完善的工作流编排工具

总结一下：

- **LangGraph**关注的是工作流的编排，你的工作流甚至可以完全与LLM无关，与Agent无关。如果你利用LangGraph开发Agent，可以自由的控制Agent工作流中的每一个细节

- **LangChain**则是提供的统一API，底层基于LangGraph实现Agent，简化了Agent的开发。

掌握了LangGraph，你就掌握了开发复杂度高、可靠性高的Agent的能力，迈入了高级AI应用开发工程师的门槛。

接下来，我们就正式开始学习LangGraph，我们会分为以下几个部分：

1. LangGraph的基本概念和快速入门

1. LangGraph中常见的WorkFlow形式

1. 培养LangGraph思维模式

---

## 第1节. 基本概念和入门

### 1. 基本概念

正如前面所述，LangGraph中包含的核心组件有：

- **状态（State）**：整个工作流中流转的数据

- **节点（Node）**：工作流中的关键工作代码，主要任务是加工处理State中的数据，用函数来定义

- **边（Edge）**：也可以理解为线，就是把各个节点连接起来的路径，是控制工作流走向的关键

由Edge和Node组成了一个按照固定顺序执行的工作流，也就是**图（Graph**）。图的输入是State、输出是加工后的State，所以图就是加工处理State的"流水线"。

> [白板/画板内容] (token: WqcVdtUK2opxk7xBXqac4UVQnvf)

接下来，我们就逐个学习LangGraph中的这些概念，以及如何利用这些组件构成**图（Graph）**.

#### 1.1 节点（Node）

节点（Node）是图（Graph）的执行单元，用来加工处理数据，多数情况下节点就是一个Python函数。


格式如下：

```Python

def my_node(state: State) -> dict:
    # 从state读取数据
    # 执行计算逻辑（可以调用LLM、工具、数据库等）
    # 返回state的更新部分（dict）
    return {"age": new_value}
```

- **输入**：State对象

- **输出**：也是State对象，但是可以用字典，只包含要更新到state的字段（部分更新）

- **可以做什么**：调用LLM、执行工具、读数据库、文件操作，最终把结果更新到State中...

需要注意的是，LangGraph有两个默认的Node是无需定义的，可以直接使用：

- **START**：开始节点，也是入口

- **END**：结束节点，也是出口

例如，我们要做一个简单的Graph：

start(用户输入name) --> 节点1（生成问候语）--> 节点2（把问候语转大写）--> end

```Python
from typing import TypedDict
```

生成的图结构：

> [白板/画板内容] (token: KAondcCilocbkgxbHWtcMLQlnAh)

总结，创建Graph的一般流程是这样的：

- 定义State，定义数据结构

- 定义Node，定义数据处理逻辑

- 创建Graph，分为几部分：

- 执行Graph

#### 1.2 State（状态）

State是图的共享内存数据，贯穿整个执行过程。

- State定义Graph中的数据字段（输入的字段、运行的结果）

- 每个Node都可以获取State数据、更新的State数据（返回要更新的字段值即可）

而Node返回数据后State的更新处理方式取决于**`Reducers`**，而且State中的每个字段都有自己的`Reducer`。

reducer:本质就是一个函数，这个函数接收原本的数据，和新数据，返回整合后的结果

比如：

```Python
greeting = 'hello, Jack'

def reducer(old, new):
```

##### 1.2.1 默认Reducer

需要注意的是：如果字段没有指定Reducer，那**默认的Reducer行为就是覆盖**。

也就是说：如果多个Node都更新了State中的同一个字段，默认情况下后执行的Node将覆盖前面Node的更新结果。例如：

```Python
from typing import TypedDict
from langgraph.constants import START, END
```

上案例中每个Node都在修改State的`val`字段，但最终`val`只记录了最后一次的值。这就是默认的Reducer行为。

结果：

```Python
node_1 is receiving val: default
node_2 is receiving val: node_1
{'val': 'node_2'}
```

##### 1.2.2 自定义Reducer

通过前面的例子我们就能看到默认的Reducer效果了。

如果我们不希望覆盖，则需要自定义Reducer，需要用到Annotated来定义：

- `Annotated[type, reducer]`

```Python
from operator import add
from typing import Annotated, TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


# =================1. 定义State数据结构和Reducer=================
class CustomReducerState(TypedDict):
```

结果：

```Python

```

可以看到，尽管Node中每次都是直接修改State的nodes字段，但最终由于使用了add这个Reducer，它会把nodes的初始值与新值拼接，类似于：

```Python
nodes = []
nodes = nodes + ['node1'] + ['node2']
```

#### 1.3 Edge（边）

Edge定义节点的执行顺序。LangGraph提供两种Edge：

| 边类型 | 方法 | 说明 |
| --- | --- | --- |
| 普通边（Normal Edge） | add_edge(node_a, node_b) | 固定流向：A执行完一定到B |
| 条件边（Conditional Edge） | add_conditional_edges(node_a, router, mapping) | router的返回值决定下一节点 |

示意图：

> [白板/画板内容] (token: M1jBdhnGQoQOC5xbD9UccC20n7f)

##### 1.3.1 Normal Edge

普通边连接的两个节点流向是固定的，前一个节点执行完一定会执行后一个节点。不同之处在于节点之间是串行还是并行：

- 串行: 整个工作流只有一条路径，路径中的节点按照顺序依次执行

- 并行: 整个工作流由多条路径，不同路径可以同时执行

上节我们演示的就是串行方式，本节课我们再来看看并行方式，结构如图：

```Python
      ┌──>  b ───┐
a ────┤          ├────> d
      └──>  c ───┘
```

代码：

```Python
from operator import add
from typing import Annotated, TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph
```

最终生成的Graph如图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Ie0RbEcoyoeYjBxRn2pcsTsvnAd/)

运行结果：

```SQL

```

能发现，b和c确实是并行运行，因为node_b和node_c接收到了相同的参数。

 
> 注意：如果并行节点同时修改了State的同一个字段，那么这个字段就必须有自定义的Reducer来处理，否则就会出现冲突，会报错！

##### 1.3.2 Conditional Edge

条件边的添加方式如下：

```Python
add_conditional_edges(node_a, router, mapping)
```

接收三个参数：

- `node_a`：是当前节点（这条边开始）

- `router`：路由函数，逻辑自定义，它的返回值默认就是下个节点的名字。

- `mapping`：如果`router`返回值与下个节点名不一样，可以用mapping定义返回值与下个节点名字的映射关系

例如，我们实现一个成绩判定的Graph，用户输入一个分数，我们根据分数判断走哪个节点：

- 大于60，走**成功**节点

- 小于60，走**失败**节点

> [白板/画板内容] (token: NRT8d6dV5oD95ExOlp2c0Y32nMc)

代码：

```Python
from typing import TypedDict, Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph
```

结果：

```Python

```

图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/N1znbyc4UomRMNxy76HcxL7jnac/)

##### 1.3.3 Command实现条件分支

如果不想编写Conditional Edge，也可以在Node中直接返回下个节点信息。

但是，由于Node必须返回**对State的更新**，而没有条件边就需要返回**下个节点的名字**。函数返回值只能有一个。所以LangGraph就提供了Command API，用`Command`来指定下个节点以及要更新的字段。

示例：

```Python
def node_a(state: State) -> Command[Literal['node_b', 'node_c']]:
    # ...
    return Command(
        update={"k":"v"},
        goto="node_b"
    )
```

其中：

- `update `: 用来指定要更新的State字段

- `goto `: 用来指定下个节点

还是刚才的例子，我们用Command来实现：

```Python
from typing import TypedDict, Literal
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command


# =================1. 定义State数据结构================
```

图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/C2F8bljWuo7KIKx2M3PcQ1aXnKb/)

### 2. LangGraph构建Agent

学会了用LangGraph的基本概念，以及如何创建Graph，接下来我们就可以用LangGraph从零构建一个Agent了。

#### 2.1 基础LLM调用工作流

让我们先从最基本的LLM调用开始，先不要工具。

工作流是这样的：

> [白板/画板内容] (token: HntBd7t0Bo1sn1xP2nZciVB3nvc)

代码：

```Python
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from dotenv import load_dotenv
```

输出：

```Python
{'user_input': '你好', 'result': '你好！很高兴见到你，有什么我可以帮你的吗？'}
```

#### 2.2 消息历史

目前的Graph中，State比较简单，只记录了两个值：

- `user_input`: 一次用户输入

- `result`: 一次LLM结果

然后实际在复杂的Agent中，用户会多次与LLM交互，还有工具调用，产生大量的历史消息，这些消息都需要记录下来。因此，我们的State需要重新设计。

##### 2.2.1 自定义消息历史State

在之前学习LangChain时我们知道，Agent交互时的消息类型有很多：`SystemMessage`、`AIMessage`、`HumanMessage`、`ToolMessage`等。而这些消息都有一个共同的父类：`BaseMessage`。

因此，我们可以在State中定义一个字段，类型为**`list[BaseMessage]`**。但这还不够，由于要不断添加新的消息到这个list，所以还需要一个Reducer，而LangChain正好提供了这样的一个Reducer：**`add_message`**

示例：

```Python

```

add_message这个Reducer的效果就是不断累积消息，形成Message列表：

```Python
messages = [SystemMessage(content="你是一个热心助人的AI助手，你的名字叫武藏")]
messages = add_messages(messages, HumanMessage(content="你好，你是谁？"))
print(messages)
```

输出结果：

```Python
[SystemMessage(content='你是一个热心助人的AI助手，你的名字叫武藏', additional_kwargs={}, response_metadata={}, id='8ce1bc2c-3675-4bc6-aa46-83e43ac6e5d8'), HumanMessage(content='你好，你是谁？', additional_kwargs={}, response_metadata={}, id='89db75c5-1081-4e64-8c41-4e1930d81baa')]
```

来看一个完整示例：

```Python

```

运行结果：

```Python
{'messages': [SystemMessage(content='你扮演火箭队的武藏，以她的口吻回答问题', additional_kwargs={}, response_metadata={}, id='edec5c78-1340-4443-bc39-73dde83c03fe'), HumanMessage(content='你是谁？', additional_kwargs={}, response_metadata={}, id='9a61969b-fb3b-4f47-a507-0c380adbbd81'), AIMessage(content='我是武藏！来自火箭队的武藏！哦呵呵呵~为了捕捉稀有宝可梦，我们无所不用其极！你，小乖乖，有没有什么稀有的宝可梦可以给我看看呢？', additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 44, 'prompt_tokens': 18, 'total_tokens': 62, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 18}, 'model_provider': 'deepseek', 'model_name': 'deepseek-v4-flash', 'system_fingerprint': 'fp_8b330d02d0_prod0820_fp8_kvcache_20260402', 'id': '938440b3-93ff-468f-a8f9-fce0984fcb5c', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--019f2b2d-3017-7c20-b1a2-ffd70efcfd5b-0', tool_calls=[], invalid_tool_calls=[], usage_metadata={'input_tokens': 18, 'output_tokens': 44, 'total_tokens': 62, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}})]}
```

简化后：

```Python

```

怎么样，熟悉的味道又回来了，这个返回值是不是跟LangChain的Agent返回值一模一样呢？

没错，因为LangChain的`create_agent`正是基于LangGraph实现的~

##### 2.2.2 默认的MessageState

事实上，记录Agent消息的场景非常常见，所以LangGraph中还提供了一个默认的State，专门用来记录消息历史，叫做`MessageState`：

```Python
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

可以看到其结构与我们自定义的非常相似，只是消息类型改成了`AnyMessage`，它可以看做是所有消息的统一类型。

所以，如果只是为了记录消息历史，我们可以不用定义State了，直接用`MessageState`：

```Python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

# =================0.先初始化一个模型================
llm = init_chat_model(
    'deepseek-v4-flash',
    extra_body = {'thinking': {'type': 'disabled'}}
)
```

#### 2.3 会话记忆

有了历史并不等于有记忆，记忆必须基于**会话id**(`thread_id`)来分别管理会话历史。这就要靠LangGraph中的`Checkpointer`来实现了。

之前在LangChain中我们就是导入的LangGraph提供的`Checkpointer`来实现记忆的，这里也一样。我们用默认的`InMemorySaver`来演示。

```Python
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

# =================0.先初始化一个模型================
llm = init_chat_model(
    'deepseek-v4-flash',
    extra_body = {'thinking': {'type': 'disabled'}}
)
```

结果：

```Python
================================ Human Message =================================
```

#### 2.4 Streaming

LangGraph中的streaming方式与LangChain的Agent一样：

```Python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.constants import START, END
from langgraph.graph import MessagesState, StateGraph
from langgraph.runtime import Runtime


# ===============Step 1: 初始化模型===============
llm = init_chat_model(
```

#### 2.5 带工具调用的LLM工作流（了解）

Agent通常是绑定了工具的LLM。它没有固定的工作流，而是采用ReAct行为模式：

1. 把问题交给LLM处理（Reasoning）

1. 调用工具（Action）

1. 拿到工具结果，返回结果给LLM，判断是否能回答问题（回到1）

1. 生成响应，结束

如果我们把"调用LLM"、"工具执行"，都看做是Node，那节点的流转就是由LLM控制的，动态循环的过程。

> [白板/画板内容] (token: ZKKhdXkAxoWUURxC2KvcIK6tnie)

 
> **注意**：为了能直观看到模型输出，方便编写解析代码，本节采用Notebook开发。

以下是本节完整依赖：

```Python
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from dotenv import load_dotenv
load_dotenv()
```

##### 2.5.1 定义Tool

```Python
# ====== Step 1: 准备工具 ======
# 工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    weather_data = {
        "北京": "晴天 25度",
        "上海": "多云 28度",
        "杭州": "小雨 22度",
    }
```

需要注意的，手动调用带工具的模型，模型会返回要执行的工具的信息，我们需要解析这些信息，拿到要调用的工具名称和参数，自己去调用。

所以，我们才在上面建立了工具name与工具的映射。将来AI返回工具信息时，我们就必须自己手动解析、调用工具了。

##### 2.5.2 定义模型，绑定工具

调用模型时，必须把工具信息一起发给模型，自己组织信息太麻烦了，所以LangChain提供了一个简化办法，就是把工具与模型绑定，将来LangChain会自动组织工具信息，帮我们发送：

```Python
from langchain.chat_models import init_chat_model
# ====== Step 2: 准备模型，绑定工具 ======
# 定义模型
model = init_chat_model(
```

##### 2.5.3 定义Node

```Python
# ====== Step 3: 定义Node ======
def tool_node(state: MessagesState):
    # 获取最后一条消息，也就是AI返回的带工具的消息
    message = state['messages'][-1]
```

##### 2.5.4 构建Graph并测试

```Python
# ====== Step 4: 构建图 ======
agent_graph = (
    StateGraph(MessagesState)
    .add_node("llm", llm_node)
    .add_node("tools", tool_node)
    .add_edge(START, "llm")
    .add_conditional_edges("llm", should_continue)
    .add_edge("tools", "llm")  # 工具执行后回到LLM，形成循环
    .compile()
)


# ====== Step 5: 调用 ======
result = agent_graph.invoke({
```

完整代码：

```Python
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.tools import tool
```

最终构成的Graph结构：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/QMcdbYX8eo19sHxQjjpc8wlwnfd/)

结果：

```Python
================================ Human Message =================================

北京和杭州今天天气怎么样？
================================== Ai Message ==================================

我来查一下北京和杭州今天的天气情况。
Tool Calls:
  get_weather (call_00_s1nnvsGmkieKfe59xIt24885)
 Call ID: call_00_s1nnvsGmkieKfe59xIt24885
  Args:
```

##### 2.5.5 预定义工具节点

大家发现了吗，自己写工具节点是不是挺繁琐的，所以LangGraph中提供了几个预定义的节点：

- `langgraph.prebuilt.ToolNode`：工具节点，内置工具执行逻辑

- `langgraph.prebuilt.tools_condition`：路由节点，判断是否要调用工具

所以，我们的代码可以简化为：

```Python
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition


# ====== Step 1: 准备工具 ======
```

#### 2.6 Runtime Context（了解）

RuntimeContext就是运行时上下文，是一次请求中可以共享的一些信息。通常用来传递一些配置信息、用户敏感信息

在LangGraph中同样支持Runtime Context功能，方式也完全一样：

- 定义ContextSchema

- 创建Graph时指定context_schema

- 使用时传递Context

示例：

```Python

```

#### 2.7 Interrupts

**中断(Interrupts)**允许您在特定Node暂停Graph执行，并在继续之前等待外部输入。利用这个特性，就能实现“**Human In The Loop**”模式，把一些重要决断交给人工来做。

当中断被触发时，LangGraph使用它的`Checkpointer`保存Graph的`State`，并无限期地等待，直到恢复执行。

你可以在Graph的任何Node调用`interrupt()`函数来中断工作。该函数接受**向调用者显示的**任何json可序列化的值。

当您准备好继续时，您可以通过使用`Command(resume={})`重新调用Graph来恢复执行，而`resume`值将成为节点内部`interrupt()`调用的返回值。

一定要注意：断点恢复必须指定`thread_id`，它是找到不同会话Checkpointer的关键。

我们以邮件助手为例：

```Python

```

接下来测试一下：

```Python

```

运行效果：

```YAML

```

继续：

```SQL
stream = hitl_graph.stream_events(
    {"messages": [HumanMessage("帮我回复Jane，告诉她很高兴她能来，问问具体时间")]},
```

会发现运行后没有任何输出，这是因为调用发邮件工具时被interrupt中止了。

我们可以通过`stream.interrupts`来获取中断信息：

```Python
interrupt_info = None
if stream.interrupted:
```

结果：

```SQL

```

果然，这个输出信息就是`send_email`节点内部调用`interrupt`时传入的信息。

理论上接下来就是人机交互，让用户选择接下来的操作：approve/reject/edit了。

我们定义一个方法，用键盘录入模拟人机交互：

```Python
# 定义收集用户指令的方法
def get_human_decision(interrupt_info: dict):
    # 这里模拟前端交互，让用户确认接下来的操作
    action = input(f"""
    subject: {interrupt_info['args']['subject']}\n
    to: {interrupt_info['args']['to']}\n
    body: {interrupt_info['args']['body']}\n
    {interrupt_info['confirm']}\n
    可选操作：{interrupt_info['actions']}
    """)
```

运行后，如果我们输入approve，确认发送，会得到以下结果：

```Python

```

可以发现，整个处理过程其实就是一个循环。我们可以将上面的代码组装起来，形成一个对话的邮件助手Agent

```Python

```

完整的模拟流程结果：

```Markdown

```

#### 2.8 create_agent揭秘

至此你可能会发现：手动构建的图和 `create_agent` 行为几乎一样。这不是巧合——**`create_agent`**** 底层就是用 StateGraph 构建的**。

> **代码对比**

```Python
# 你写的代码：
agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather, calculate],
    system_prompt="你是一个有用的助手。",
    checkpointer=InMemorySaver(),
)

# 等同于底层手写的：
agent_graph = (
    StateGraph(MessagesState)
```

> **create_agent 做了什么？**

| 你的参数 | 底层转化 |
| --- | --- |
| model | 绑定到llm_node中 |
| tools | 注册到tool_node中，同时传给 llm.bind_tools() |
|  |  |
|  |  |
|  |  |
|  |  |

> **什么时候用哪种？**

| 场景 | 推荐方式 |
| --- | --- |
| 标准工具调用Agent | create_agent |
| 需要自定义Middleware | create_agent + Middleware |
| RAG、简单检索问答 | create_agent + Middleware |
| 多步骤流水线（检索→分析→总结） | 手动StateGraph |
| 条件分支（审批→执行/驳回） | 手动StateGraph |
| 并行执行（同时调用多个工具/模型） | 手动StateGraph |
| 复杂循环（多轮推理直到满足条件） | 手动StateGraph |
| 子流程嵌套 | 手动StateGraph |

**原则**：能用 `create_agent` 解决的先用它，需要自定义控制流时再用手动Graph。两者不是互斥的——你可以在Graph中嵌套 `create_agent` 作为子节点。

---

## 第2节. 常见WorkFlow

上一节学习了LangGraph的Node、Edge、State三要素，知道了如何构建Agent，但LangGraph的能力并不在于构建简单的Agent，而是用图的方式灵活编排任意复杂的工作流。

本节深入学习各种常见的**流程编排模式。**

本节将学习七种核心编排模式：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Eb2TbUUfhoNaKwxIWjgcuB1VnAg/)

左侧的两个：LLM嵌入到了预定义的工作流当中，用于你明确知道工作流程的场景。

右侧的三个：由LLM来控制工作流，更智能化。

本章会用到的依赖和工具：

```Python
from typing import Annotated, TypedDict, Literal, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, CachePolicy, interrupt, Send
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, TimeoutPolicy
from langgraph.errors import NodeError
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, ToolMessage
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool
```

### 1. 顺序执行（Prompt Chaining）

Prompt Chaining 是最基础的编排模式：**前一个LLM的输出作为后一个LLM的输入**，像流水线一样串联起来。

**典型场景**: 写作流程（大纲→初稿→润色）、翻译+校对、分步推理。

**与直接 LLM 调用的区别**: 顺序链可以在步骤之间加入**验证门控（Gate）**——如果某步质量不达标就退回重做，而不是一路到底。

比如，一个写笑话的工作流。

示意图：

> [白板/画板内容] (token: QbN4dEMIYoPawmx015LcWUa4n5e)

代码示例：

```Python

```

效果：

```Python

```

### 2. 并行执行（Parallelization）

通过并行化，可以让多个LLM同时处理一个任务，有两种不同的实现方式和作用：

- 提高效率: 将任务拆分为子任务，让多个LLM同时运行独立的子任务来完成，再汇总结果

- 提高准确度: 多个LLM运行相同的任务产生不同的输出，再合并优化结果

> [白板/画板内容] (token: JKu7dWt2Ro6dYlxUhIsc0ZQPnQg)

比如，我需要同时基于某个话题写三种题材：

- 笑话

- 故事

- 诗歌

并行执行的话效率会更高，示例代码：

```Python

```

结构图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/FkvsbIWwaojvJmx2loccuAFsnBg/)

测试：

```Python
# Invoke
state = parallel_workflow.invoke({"topic": "程序员"})
print(state["combined_output"])
```

结果：

```SQL
Here's a story, joke, and poem about 程序员!

STORY:
三小时前煮的最后一杯咖啡早已凉透，陈未却浑然不觉。显示器在昏暗的房间里亮着，冷白的光映在他那张两天没见太阳的脸上。
```

### 3. 路由（Routing)

Routing工作流处理用户输入，然后将它们引导到上下文相关的任务Node。这允许您为复杂的任务定义专门的流。例如：

- 智能电商客服，首先识别问题的类型，然后将请求路由到 售前咨询、退款、退货等对应的Node。

- 知识问答机器人，首先识别问题知识领域，然后将请求路由到不同领域对应的专业Node上。

所以路由本质上也是条件分支，让Graph根据问题动态选择下一节点。这是Agent智能决策的核心机制。

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/C1SfbNBZOoDOk0xcNlrcWKy2ned/)

例如，我们写三个**工作节点**：

- 天气节点：可以查询天气

- 新闻节点：可以查询新闻

- 翻译节点：可以翻译

然后写一个**意图识别节点**，根据用户意图路由到对应的工作节点，执行对应任务：

```Python

```

结构图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/DiazbsBnIo6ZbexrNvXco1Yknbh/)

测试：

```Python

# 测试三种意图
for q in ["今天北京天气如何？", "翻译hello", "你好吗？"]:
    r = routing_graph.invoke({"query": q, "intent": "", "result": ""})
    print(f"'{q}' -> intent={r['intent']} -> {r['result']}")
```

运行结果：

```Python

```

### 4. Orchestrator-Worker + Send API

**Orchestrator-Worker**，顾名思义，**编排器—工作者**模型。分为三个步骤：

- **编排器（Orchestrator）**：将任务分解为子任务，将子任务分配给worker

- **工作节点（Worker）**：并行执行任务

- **合成器（Synthesizer）**：将worker的输出汇总为最终结果

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/RLvNbK2OJop8hCxBDl3cy0ydnPe/)

这听起来似乎与之前讲的并行模式很像，没错这确实也属于并行模式。但不同之处在于：

- 普通并行工作流：任务是需要手动拆分的，且子任务数量是固定的

- Orchestrator-Worker：是用模型**动态的将复杂任务拆分为多个并行子任务**，子任务数量不确定

#### 4.1 Send API

Orchestrator-Worker模式提供了更高的灵活性，通常用于无法像并行化那样预先定义子任务的场景。例如：AI编程助手、深度研究助手

我们以深度研究助手（DeepResearcher）为例，它可以根据你指定的话题写出专业报告：

- 首先，需要根据话题生成报告大纲，细分出多个章节，定好每个章节主题

- 然后，把每个章节作为一个子任务，交给多个工作进程并行执行

- 最后，汇总所有子任务生成的章节，得到完整的研究报告

但是，这里有一个很严重的问题：

> 子任务的数量是不确定的，因此Node没办法提前定义好，那该如何设计Graph？

不用担心，LangGraph 对此提供了内置支持。通过 **Send** API，你可以动态创建工作节点并向它们发送特定的输入。每个工作节点都有自己的State，而且所有工作节点的输出都会被写入一个共享的State 字段中，编排器图可以访问该字段。这使得编排器能够获取所有工作节点的输出，并将其整合成最终输出。

在深度研究助手案例中，我们可以遍历计划好的章节列表，并使用 Send API 将每个章节发送给对应的工作节点。

#### 4.2 DeepResearcher示例

接下来就开发一个用于生成报告的深度研究助手：DeepResearcher

##### 4.2.1 结构化输出模型

**首先**，我们定义模型类，作为结构化输出的约束，让LLM按照固定格式生成拆分的章节列表：

```Python
from typing import Annotated, List
```

##### 4.2.2 State

接着，是State，工作过程中产生的章节信息和报告信息：

```Python
# 编排者的 State ，记录完整信息
```

##### 4.2.3 Normal Node

然后，是Node，包括

- 编排者节点：Orchestrator

- 工作节点：Worker

- 合成器节点：Synthesizer

```Python
# 编排者节点
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    # 通过SystemPrompt设定和结构化输出绑定，让LLM作为orchestrator
```

##### 4.2.4 派发任务的Node

接下来是关键的**任务派发**节点，我们需要遍历由Orchestrator计划好的章节列表，并使用 Send API 将每个章节发送给对应的工作节点。

```Python
from langgraph.types import Send

# Conditional edge Node，根据编排者安排的子任务创建llm_call工作者，每个工作者编写报告的一个部分
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # 通过Send() API分发任务给Worker，让Worker并行写报告的不同章节
    return [Send("worker", {"section": s}) for s in state["sections"]]
```

注意Send接收的两个参数：

- `worker`：工作节点的名字

- `{"section": s}`: 是传入的State，这里是章节信息

##### 4.2.5 创建Graph

最后，就是组织Graph了：

```Python
# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add the nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
```

结构图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/Ra3dbSLBjoYOCDxLhkucF9k9nTR/)

这个结构图看起来很奇怪，只有一个worker，但其实worker的数量是动态的，由报告规划的章节数量决定。

##### 4.2.6 测试

```Python
# Invoke
```

最后的结果：

```Markdown
# 引言
在人工智能的快速发展浪潮中，大型语言模型（LLM）已成为最具变革性的技术之一。从早期的统计语言模型到基于深度神经网络的预训练模型，LLM通过在海量文本数据上的自监督学习，掌握了丰富的语言知识和推理能力。支撑这一进步的核心原则之一便是缩放定律（Scaling Laws）。该定律揭示了模型性能与模型规模（参数数量）、数据规模（训练令牌数）以及计算资源之间的幂律关系，表明在合理范围内，增加模型大小和数据量能够带来可预见的性能提升。
```

### 5. Evaluator-Optimizer（评估-优化循环）

Evaluator-Optimizer 是提升Agent响应质量的重要模式

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/YsVTbflapoXRL1xZKvkcEe0anff/)

两个核心组件：

- **生成器（Generator）：**产出内容

- **评估器（Evaluator）：**打分/给出反馈 --> 如果不达标就带着反馈重新生成

形成一个质量提升的闭环。

与普通循环的区别：循环自带"质量门"，只在质量不达标时才重复，避免了无谓的循环。

**典型场景**: 文案优化（生成→评分→改写）、代码审查（生成→检查→修复）、翻译质量提升。

示例，我们写一个生成广告语的工作流：

```Python
# Evaluator-Optimizer: 生成广告语 → 评估 → 不达标则带着反馈重写
from typing import Literal

class AdState(TypedDict):
    product: str        # 产品名称
    slogan: str         # 当前广告语
    feedback: str       # 评估反馈
```

结构图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/M9mcb1mUSoDDYAxeF45cXnJ7nsc/)

测试：

```Python
print("=== Evaluator-Optimizer: 广告语生成 ===\n")
r = eval_opt_graph.invoke({
    "product": "黑马程序员", "slogan": "", "feedback": "", "grade": "", "iteration": 0
})
print(f"\n最终广告语: {r['slogan']}")
print(f"总共迭代: {r['iteration']} 轮")
```

运行结果：

```Markdown
=== Evaluator-Optimizer: 广告语生成 ===
```

### 6. 子图嵌套（Subgraphs）

复杂的工作流太过庞大不利于维护，此时可以拆分为多层子图。LangGraph 支持将 **已编译的子图作为父图的一个Node** 直接使用。

包括两种使用方式：

| 方式 | API | 特点 |
| --- | --- | --- |
| 直接嵌入 | add_node("name", compiled_subgraph) | 子图作为图节点，checkpoint与父图集成，interrupt自动向上冒泡 |
| 包装调用 | 在普通node函数内 subgraph.invoke() | 灵活的状态映射，需手动传递数据 |

若采用嵌入模式，且父图的State与子图的State共享某些key时：

- **父→子**: 父图自动将共享key的值传给子图（input projection）

- **子→父**: 子图返回的key如果父图也有，则按**父图的reducer**写回

如果父子图State完全不同，需要用包装调用方式手动映射。

#### 6.1 直接嵌入

我们做一个舆情检测类的工作流

- 父图：负责收集数据、汇总结果

- 子图：负责情感分析、关键词提取

以下是示例代码：

首先State：

```Python
# ========== 方式1: 直接嵌入 compiled subgraph ==========
# 子图和父图共享 State key，父图自动完成状态映射

# ----- 父图: 将来直接把子图嵌入作为节点 -----
class ParentState(TypedDict):
    input_text: str     # 可共享，父调用子时直接会传递给子图
    result: str         # 子返回结果时，会直接返回给父图
    preprocessed: str
    final_output: str

# ----- 子图: 情感分析子流程（独立编译）-----
```

然后是子图（Subgraph）：

```Python
# ========== 方式1: 直接嵌入 compiled subgraph ==========

def detect_sentiment(state: AnalysisSubState):
    """节点1: 检测情感"""
    text = state["input_text"]
```

接着是父图：

```Python

```

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/SisUbtsBUocXlmxgvU8cSQGAnEf/)

结构：

可以看到，子图的名字是：`analysis_subgraph`，子图成为了父的节点

测试：

```Python
# 测试：直接嵌入子图的父图
print("=== 测试直接嵌入子图 ===\n")
r = parent_graph.invoke({
    "input_text": "这个产品真好用，我很喜欢！",
    "preprocessed": "",
    "result": "",
    "final_output": ""
})
print(f"\n最终输出: {r['final_output']}")
```

输出：

```Markdown
=== 测试直接嵌入子图 ===

  [父图-预处理] 收到: 这个产品真好用，我很喜欢！...
    [子图-关键词] keywords=这个产品真好用，我很喜欢！
    [子图-情感分析] sentiment=正面
    [子图-合并结果] result=..
  [父图-后处理] 子图结果: 情感:正面, 关键词:这个产品真好用，我很喜欢！

最终输出: 处理完成: 情感:正面, 关键词:这个产品真好用，我很喜欢！
```

#### 6.2 包装调用

还是刚才的案例，但我们采用包装调用的方式，这样一来，不管State字段是否一样，父子图之间就无法共享数据，必须手动调用，手动传递。

子图保持不变，此处略。

接着是父图，这一次我们故意将父图的State字段改成与子图不一致：

```Python
# ----- 父图: 直接把子图嵌入作为节点 -----
```

结构：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/R6w9bDD8lonrLExTJyxcTdr7n0b/)

可以看到，子图的名字已经是`wrapper_node`了，不是子图原本的名字，这就是包装调用。

测试：

```Python

```

输出：

```Markdown

```

### 7. 总结

##### 七种编排模式速查

| 模式 | API | 原理 | 典型场景 |
| --- | --- | --- | --- |
| Prompt Chaining | add_edge 串联 + add_conditional_edges 做门控 | 每个LLM输出作下一个LLM输入，中间可插入质检 | 写作流程、翻译+校对、分步推理 |
| Routing | add_conditional_edges(node, router, mapping) | router函数返回节点名，mapping映射到实际节点 | 意图路由、权限分流、LLM工具调用判断 |
| Parallelization | 多条add_edge指向不同目标 | 节点间无依赖即可并行，需要reducer合并结果 | 同时查多个数据源、并行LLM调用 |
| Orchestrator-Worker | add_conditional_edges + Send API | 动态生成Send列表，并行分发到同一节点不同State | 批量处理、子问题分解、分布式搜索 |
| Evaluator-Optimizer | 条件边回环 + 评估器门控 | 生成→评估→不达标带反馈重写，达标则退出 |  |
|  |  |  |  |

##### 核心要点

1. **Prompt Chaining 是最简单的编排** — 串联LLM调用，中间加入门控检查，形成可验证的流水线

1. **路由是智能的基础** — Agent通过`add_conditional_edges`根据State动态选择路径

1. **并行提升性能** — 无依赖的节点可并行执行，用`operator.add`等reducer合并结果

1. **Orchestrator-Worker** — 动态任务拆分与并行执行，是智能与效率的结合，需要Send API

1. **Evaluator-Optimizer保障质量** — 生成→评估→反馈→重写循环，直到达标

1. **子图实现分层设计** — 复杂流程拆为独立子图，`add_node(graph)`直接嵌入，共享State自动映射

##### 模式选择指南

- 确定步骤顺序 → Prompt Chaining

- 存在不同领域的Node → 智能路由(Routing)

- 多个独立任务加速 → 并行执行(Parallelization)

- 复杂任务动态拆分和性能提升 → Orchestrator-Worker(Send)

- 需要迭代优化质量 → Evaluator-Optimizer

- 流程太长需要拆解 → 子图嵌套

---

## 第3节. Think In LangGraph

前两节我们学习了 LangGraph 的三要素和七种编排模式，它们解决的是 **「怎么做」** 的问题。

本节换个角度，从零构建一个客户支持 Agent 的完整流程，理解 LangGraph 背后的思维框架——**「怎么想」**。

LangGraph 构建 Agent 有5个核心阶段：

- 将任务拆解为一个个离散的节点，Node

- 明确每个节点的具体任务

- 设计节点中流转的数据格式，State

- 编写节点函数，做好异常处理

- 连接节点，形成图，Graph

传统 Prompt Chaining 是线性的 A→B→C，但真实世界的 Agent 是**有分支、有回路、有人工介入**的图结构。

要正确创建这个图就必须学会LangGraph的思维，接下来我们就一个客户支持的EmailAgent为例来学习如何**Thinking in LangGraph**.

### 0. 客户支持 Agent：需求分析

假设我们要构建一个自动处理客服邮件的 Agent，它需要：

- 读取收件箱中的新邮件

- 按紧急程度和主题分类

- 搜索知识库文档寻找答案

- 撰写回复草稿

- 复杂问题升级给人工处理

- 必要时安排跟进

以下五个场景代表了不同类型的问题：

| 场景 | 类型 | 处理策略 |
| --- | --- | --- |
| 密码重置 | 简单问题 | 查文档 → 自动回复 |
| PDF导出崩溃 | Bug | 创建工单 → 人工审核 |
| 重复扣款 | 紧急账单 | 直接人工处理 |
| 暗黑模式功能 | 功能请求 | 查文档 → 标记feature |
| API间歇性504 | 复杂技术 | 直接人工处理 |

**核心观察**：不同的问题类型走不同的处理路径 —— 这就是我们需要**图**而不是链的原因。

### 1. 设计工作流

首先不着急写代码，先梳理业务，把业务流程用图画出来：

> [白板/画板内容] (token: UmR2dViyIoCRfGxNAFicULkwnFg)

七个节点，虚线表示条件路径。不过，我们不定义条件边，而是把决策定义在节点内部。

每个节点只做一件事：

| 节点 | 职责 | 决策？ |
| --- | --- | --- |
| 读取邮件 | 解析邮件内容和发件人 | 否 |
| 意图识别 | LLM分类意图+紧急度，决定下一步路径 | 是 |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

### 2. 理清节点的职责

接下来还不是写代码，而是先明确每个节点的**输入**、**输出**、**失败策略**。这决定了你的 State 设计和Node的错误处理架构。而这些跟节点的职责类型有关。

LangGraph 中有节点按照职责可以分为四类：

- **LLM节点**：主要职责是理解、分析、生成

- **Data节点**：主要职责是检索外部数据

- **Action节点**：职责是执行外部操作

- **Human节点**：职责是负责人机交互

接下来， 我们看看本例中的节点分别属于哪种，各自的输入和输出及失败策略。

#### 2.1 **LLM 节点**

| 节点 | 静态上下文（Prompt模板） | 动态上下文（State） | 期望输出 |
| --- | --- | --- | --- |
| 意图识别 | 分类类别、紧急度定义 | 邮件内容、发件人 | 结构化分类结果 + 路由 |
| 生成草稿 | 语气指南、公司政策 | 分类结果、搜索结果、客户历史 | 专业回复草稿 |

#### 2.2 **Data 节点**

| 节点 | 查询参数 | 重试策略 | 缓存 |
| --- | --- | --- | --- |
| 文档检索 | 从意图和主题构建查询 | 快速失败重试 | 常用查询永久缓存 |
|  |  |  |  |

#### 2.3 **Action 节点**

| 节点 | 执行时机 | 重试策略 | 幂等性 |
| --- | --- | --- | --- |
| 发送邮件 | 审批之后 | 网络问题快速失败重试 | 不缓存，每次唯一 |
| BUG跟踪 | 意图为bug时 | 必须重试，不能丢失BUG报告 | 返回Ticket ID |

#### 2.4 **Human 节点**

| 节点 | 呈现给人工的上下文 | 预期输入 | 触发条件 |
| --- | --- | --- | --- |
| 人工确认 | 原邮件 + 草稿 + 紧急度 + 分类 | 是否批准或修改后的回复 | 高紧急度、复杂问题 |

### 3. 设计 State

State 是**所有节点共享的"工作笔记本"**。怎么判断一个字段要不要放 State？

> **原则：如果下游节点需要用它做决策或处理，就入 State。如果能从其他字段推导出来，就不存。**

对于我们的Email Agent，我们需要存入State的信息包括：

- 原始邮件及发件人信息（之后无法重建）

- 分类结果（被后续多个节点所需）

- 搜索结果和客户数据（重新获取成本较高）

- 回复邮件的草稿（需在评审期间展示）

因此，其State要这样定义：

```Python
from typing import TypedDict, Literal
```

### 4. 构建节点

分析好节点职责，接下来就是创建节点了。节点就是一个 Python 函数，接收当前 State，返回 State 更新。

不过，要特别注意的是工作流中的任何一个节点异常都可能导致整个工作流崩溃！所以，必须妥善处理节点中出现的异常。

不同的异常类型往往有不同的处理策略：

| 错误类型 | 谁来修复 | 处理策略 |
| --- | --- | --- |
| 短期异常 (网络问题、限流) | 系统自修复 | 多次重试：异常属于临时现象，多次重试就可能恢复正常 |
| LLM调用异常 (工具执行失败,数据格式问题) | LLM | 把错误信息写入State，再次交给LLM来处理 |
| 用户输入错误 (信息缺失、格式错误) | 用户 | 基于 interrupt()，人工修复 |
| 可恢复的错误 | 开发者 | 重试次数耗尽后，通过定义的error_handler来处理 |
| 其它预料之外的错误 | 开发者 | 直接抛出 |

接下来，我们分别定义各个节点：

#### 4.1 邮件读取与意图识别

**意图识别**后需要判断分支走向：

- 简单问题：**检索文档**

- BUG报告：**BUG跟踪**

- 其它：**人工审核**（此处我们改一下，先去生成草稿，再人工审核，让人工在草稿基础上去改，省事）

这里我们不用条件边，而是用Command来指定下个节点。

```Python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, RetryPolicy

# 给LLM绑定结构化输出，做为路由决策
classifier_llm = llm.with_structured_output(EmailClassification)

def read_email(state: EmailAgentState) -> dict:
    """节点1: 解析邮件——总是走到下一个固定节点"""
    # 真实业务可以连接邮件服务器，读取信息
    print(f"  [Read] 收到邮件: {state['email_id']}")
    return {}
```

#### 4.2 文档搜索与BUG跟踪

**文档搜索**和**BUG跟踪**都可以看做是具体的工作节点，完成后都需要跳转到**生成草稿**的节点。

```Python
# 文档搜索/BUG工单处理完成后直接跳去draft_response，直接用Command跳转

def search_documentation(state: EmailAgentState) ->Command[Literal['draft_response']]:
    """节点3: 搜索知识库（带重试策略）"""
    classification = state.get('classification', {})
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"
    print(f"  [Search] 查询: {query}")

    # 模拟知识库搜索，实际开发可以改为RAG
    search_results = [
        "密码重置: 登录 -> 设置 -> 安全 -> 修改密码",
        "密码至少12位，包含大小写和特殊字符",
```

#### 4.3 生成草稿

**生成草稿**后同样需要判断分支走向：

- 如果是*对于紧急任务、复杂任务、账单相关，都走***人工审核**

- 其它类型：**直接发送**

```Python
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """节点5: LLM生成回复草稿 + 判断是否需要人工审核"""
    classification = state.get('classification', {})

    # 格式化上下文（在节点内部格式化，不污染State）
    context_parts = []
    if state.get('search_results'):
        context_parts.append("文档:\n" + "\n".join(f"- {r}" for r in state['search_results']))
    if state.get('customer_history'):
        context_parts.append(f"工单: {state['customer_history']}")

    prompt = f"""起草一封回复邮件：
客户: {state['sender_email']}
```

#### 4.4 人工确认

人工确认需要与前端约定好交互方式。这里我们返回的信息包括：

- 原始邮件信息：发件人、原邮件内容

- 邮件意图信息：意图、紧急情况

- 回复邮件草稿

- 可选的行动：用来提示用户接下来的操作，这里我们简化为两个操作

```Python
from langgraph.types import interrupt  # interrupt 在 langgraph.types 中

def human_review(state: EmailAgentState):
    """节点6: 人工审核"""
    classification = state.get('classification', {})

    human_decision = interrupt({
```

### 5. 构建图（最小边原则）

由于节点内部已经通过 `Command` 声明了路由，图的边只需要定义**固定跳转**和**起点/终点**，结构非常简洁。

```Python

```

结构如图：

![图片](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/v2/cover/SWwybFhLwo2is2x2fKqc2Drfnjd/)

### 6. 测试

我们分别测试三种场景：

- 普通问题

- BUG跟踪

- 紧急问题

#### 6.1 普通问题

```Python
# 测试三个不同场景
config_base = {"configurable": {"thread_id": "test"}}

print("=== 场景1: 密码重置（查文档 → 自动回复）===\n")
config1 = {"configurable": {"thread_id": "t1"}}
r1 = app.invoke({
    "email_content": "Hi，我忘记登录密码了，如何重置？",
    "sender_email": "jack@example.com", "email_id": "email-001",
    "classification": None, "search_results": None,
    "customer_history": None, "draft_response": None
}, config1)
```

运行结果：

```Markdown
=== 场景1: 密码重置（查文档 → 自动回复）===

  [Read] 收到邮件: email-001
  [Classify] intent=question, urgency=low
  [Classify] -> 路由到: search_documentation
  [Search] 查询: question 密码重置
  [Draft] 回复已生成 (155 字符)
  [Draft] -> 路由到: send_reply
  [Send] 发送回复邮件给jack@example.com成功...

**主题：** 关于密码重置的指引  

Hi Jack，  
```

#### 6.2 BUG报告

```Python
print(f"\n=== 场景2: Bug报告（建工单 → 回复）===\n")
config2 = {"configurable": {"thread_id": "t2"}}
r2 = app.invoke({
    "email_content": "PDF导出功能每次都会崩溃，急需修复！",
    "sender_email": "jane@corp.com", "email_id": "email-002",
    "classification": None, "search_results": None,
    "customer_history": None, "draft_response": None, "messages": None
}, config2)

print(f"  回复: {r2['draft_response']}...")
```

结果：

```Markdown

```

#### 6.3 紧急问题

```Python

print(f"\n=== 场景3: 账单问题（直接升级人工审核）===\n")
config3 = {"configurable": {"thread_id": "t3"}}
```

结果：

```Markdown

=== 场景3: 账单问题（直接升级人工审核）===

  [Read] 收到邮件: vip-email-003
  [Classify] intent=billing, urgency=critical
```

没有打印出最终回复的邮件，这是因为需要人工审核。

我们模拟一个人工审核，比如approve：

```Python
# 人工确认，比如approve
r4 = app.invoke(
    Command(resume={
        "action": "edit",
        "edited_response": """
```

然后就能拿到回复邮件了：

```Markdown

```
