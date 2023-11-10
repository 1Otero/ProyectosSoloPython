import subprocess 

#bodyTerm= subprocess.run(["ls", "-l"], stdout= subprocess.PIPE, text=True, shell=True)
bodyTerm= subprocess.run(["ls", "-l"])

print(bodyTerm)
