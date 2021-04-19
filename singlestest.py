import subprocess
import glob

lr = 0.01
mom = 0.01
weight_init = 'xavier_uniform'

all_paths = glob.glob("data/*")
task_paths = []
for p in all_paths:
    if 'stats' not in p and 'Comp' in p:
        task_paths.append(p)

for repeat in range(3):
    for p in task_paths:
        subprocess.run(["python", "singletasktest.py", p,
                        str(lr), str(mom), weight_init, str(repeat)])
