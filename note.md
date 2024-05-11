
# test.json 的说明

这个JSON文件描述了一个图（Graph），具体内容包括：

- **n_v**: 图中顶点的数量（100个顶点）。
- **d**: 图的度（Degree），这里是9，意味着每个顶点连接9条边。
- **type**: 图的类型（"regular"），表示这是一个规则图，每个顶点都有相同数目的边。
- **IsWeighted**: 图是否加权（0表示未加权）。
- **n_e**: 图中边的数量（450条边）。
- **edges**: 边的列表，每个条目包含两个顶点编号和权重（这里权重都为1，即使IsWeighted为0，表示所有边等权重）。

例如，边列表的第一条 `[36, 53, 1]` 表示顶点36和顶点53之间有一条权重为1的边。整个`edges`数组描述了图中所有的边，这些信息可以用来可视化图，或者进行图论的计算和分析。

# 如何将上图plot出来

要从一个名为`test.json`的文件中自动读取`edges`数据并使用它们创建图，你可以使用Python的`json`模块来加载JSON数据。这里是修改后的代码，展示如何从文件读取`edges`并绘制图：

1. 确保`test.json`文件包含了你提供的JSON数据，并且它位于你的Python脚本可以访问的目录下。

2. 使用以下Python代码来读取JSON文件并绘制图：

```python
import json
import networkx as nx
import matplotlib.pyplot as plt

# 读取JSON文件
with open('test.json', 'r') as file:
    data = json.load(file)

# 创建一个新的图
G = nx.Graph()

# 添加边
edges = data['edges']
for edge in edges:
    G.add_edge(edge[0], edge[1], weight=edge[2])

# 绘制图
pos = nx.spring_layout(G)  # 使用spring布局
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='#909090', node_size=500, font_size=8, font_weight='bold')
plt.show()
```

这段代码先打开并读取`test.json`文件，然后从中获取`edges`数据。接下来，它将这些边添加到图中，并使用与之前相同的方法绘制图。

确保`test.json`文件与Python脚本在同一文件夹中，或者提供正确的文件路径。这将自动从你的JSON文件中提取边数据并绘制图。