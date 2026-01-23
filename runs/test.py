
all_events = [0, 1, 2, 3, 4, 0, 5, 3, 2, 1, 2, 0, 2, 5, 1, 2, 3, 4]  # 0: episode end, 1: episode start, 2: free space, 3: handle move, 4: hinge move, 5: episode success
print(all_events)

between_eps = False
filtered_events = []
for event in all_events:
    if event == 0:
        between_eps = True 
    elif event == 1 and between_eps:
        between_eps = False

    if between_eps and event != 0:
        continue

    filtered_events.append(event)

print(filtered_events)