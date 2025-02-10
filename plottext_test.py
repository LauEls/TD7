import plotext as plt
import numpy as np

plt.theme("pro")
plt.subplots(1, 3)


y_sin = []
y_cos = []

plt.subplot(1, 1).title("Sin Plot")
# plt.subplot(1, 1).xlim(0, 100)
plt.subplot(1, 1).ylim(-1, 1)

plt.subplot(1, 2).title("Cos Plot")
plt.subplot(1, 2).xlim(0, 100)
plt.subplot(1, 2).ylim(-1, 1)





for i in range(100):
    # print(f"i: {i}")
    
    y_sin.append(np.sin(i/10))
    y_cos.append(np.cos(i/10))
    # y = plt.sin()
    plt.subplot(1, 1).plot(y_sin)
    plt.subplot(1, 2).plot(y_cos)

    plt.subplot(1,3).indicator(i, "Episode")
    # plt.plot(y_sin)
    # plt.text(f"i: {i}",x=1,y=1, color="red")


    plt.subplot(1,1).xlim(0, len(y_sin))
    plt.show()