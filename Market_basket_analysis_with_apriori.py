
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/MyDrive/groceries.csv'  # Replace with your actual path
df = pd.read_csv(file_path, on_bad_lines='skip',engine='python')

df.head(2)

print("Column's unique count")
for i in df.columns:
  print(i,":",df[i].nunique())

basket = df.groupby(['Member_number','itemDescription'])['Date'].count().unstack().reset_index().fillna(0).set_index('Member_number')

basket.head(10)

def encode_units(x):
  if x <=0:
    return 0
  if x >=1:
    return 1

basket = basket.map(encode_units)

wine_df = basket.loc[basket['white wine']==1]

from mlxtend.frequent_patterns import apriori, association_rules

# Applying apriori: finding itemsets
frquent_itemsets = apriori(wine_df, min_support = 0.5, use_colnames=True)

frquent_itemsets #0.5 indicates 50% of the transactions contains that item.

wine_rules = association_rules(frquent_itemsets,metric='lift',min_threshold=0.6)

wine_rules

import matplotlib.pyplot as plt
import networkx as nx
# Scatter plot of confidence vs lift
plt.figure(figsize=(8,6))
plt.scatter(wine_rules['confidence'], wine_rules['lift'], alpha=0.7, color='b')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Confidence vs Lift in Association Rules')
plt.grid()
plt.show()
# Visualizing association rules as a network graph
G = nx.DiGraph()
for _, row in wine_rules.iterrows():
    G.add_edge(tuple(row['antecedents']), tuple(row['consequents']), weight=row['confidence'])
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10)
edge_labels = {(tuple(row['antecedents']), tuple(row['consequents'])): f"{row['confidence']:.2f}"
               for _, row in wine_rules.iterrows()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Association Rules Network")
plt.show()

basket[['white wine','whole milk']].where(basket['white wine']==1).notna().sum()
