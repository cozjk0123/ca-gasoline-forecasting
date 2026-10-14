import sys, nbformat
from nbclient import NotebookClient
p=sys.argv[1]; nb=nbformat.read(p,4)
c=NotebookClient(nb,timeout=int(sys.argv[2]) if len(sys.argv)>2 else 1800,kernel_name="python3",resources={"metadata":{"path":"notebooks"}})
try:
    c.execute()
finally:
    nbformat.write(nb,p)
for cell in nb.cells:
    if cell.cell_type=="code":
        for o in cell.outputs:
            if o.output_type=="stream": print(o.text[:1500])
            elif "data" in o and "text/plain" in o["data"] and "Figure" not in o["data"]["text/plain"]: print(o["data"]["text/plain"][:2000])
            elif o.output_type=="error": print("ERR",o.ename,o.evalue)
