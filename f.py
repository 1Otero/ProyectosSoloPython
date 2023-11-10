import subprocess

bodySubprocess= subprocess.run(["ls","-l"])

print(bodySubprocess)

bodySubprocess= subprocess.run("ls")

print(bodySubprocess)

