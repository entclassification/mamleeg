import task
import model
import glob


all_paths = glob.glob("data/*")
task_paths = []
for p in all_paths:
    if 'stats' not in p and 'Comp' in p:
        task_paths.append(p)

ta = task.competetask(task_paths[0], 1)

filt = model.Filter(inp_channels=22, inp_len=250, out_len=128)

for x, y in ta.train():
    filt(x)
    break
