import numpy as np 
import matplotlib.pyplot as plt 

barWidth = 0.2
# fig = plt.figure()
fig = plt.figure(figsize=(18.0, 13.0), dpi=100)

td7_demo = [0.96, 0.84] 
td7 = [0.0, 0.29] 
bc = [0.74, 0.62]
demo_replay = [0.28, 0.37] 

br1 = np.arange(len(td7_demo)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
br4 = [x + barWidth for x in br3] 

bars1 = plt.bar(br1, td7_demo, color=(0.8500, 0.3250, 0.0980), width = barWidth, 
        edgecolor ='black', label ='TD7+Demo') 
bars2 = plt.bar(br2, td7, color=(0.9290, 0.6940, 0.1250), width = barWidth, 
        edgecolor ='black', label ='TD7') 
bars3 = plt.bar(br3, bc, color=(0.4660, 0.6740, 0.1880), width = barWidth, 
        edgecolor ='black', label ='BC') 
bars4 = plt.bar(br4, demo_replay, color=(0.4940, 0.1840, 0.5560), width = barWidth, 
        edgecolor ='black', label ='Demo Replay')

# Add labels on top of each bar
def add_labels(bars, is_percentage=False):
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label = f'{height * 100:.0f}%' if i == 0 else f'{height:.2f}'
        plt.text(bar.get_x() + bar.get_width()/2.0, height, label, 
                 ha='center', va='bottom', fontsize=55)

add_labels(bars1)  # Success rate group as percentages
add_labels(bars2)
add_labels(bars3)
add_labels(bars4)

# Increase font sizes
plt.xticks([r + barWidth * 1.5 for r in range(len(td7_demo))], 
           ['Success Rate', 'Mean Reward'], fontsize=55)
plt.yticks(fontsize=55)
# plt.legend(fontsize=55)

# Remove top and right border lines
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)  # Optionally remove left border line as well

# Remove y-axis tick labels
ax.set_yticklabels([])

# Remove the entire y-axis
ax.get_yaxis().set_visible(False)

# plt.xlabel('Branch', fontweight ='bold', fontsize = 15) 
# plt.ylabel('Students passed', fontweight ='bold', fontsize = 15) 

plt.show()
