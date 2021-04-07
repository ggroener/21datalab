file = open(r"..\venv\Scripts\activate.bat","r")

lines = []
while 1:
    try:
        line = file.readline()
        if line:
            lines.append(line)
        else:
            break
    except:
        break      
file.close()

file=open(r"..\venv\Scripts\activate.bat","w")
for l in lines:
    l=l.replace("delims=:","delims=:.")
    file.write(l)
file.close()
    
    