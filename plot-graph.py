import json
import networkx as nx
import matplotlib.pyplot as plt

# 读取JSON文件
with open('data/test.json', 'r') as file:
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
