#-*- coding: utf-8 -*-
import os,io
import sys,tqdm

#uniq={"'","_","a","b","c","d","e","f","g","h","i","j","k","l",
#"m","n","o","p","q","r","s","t","u","v","w","x","y","z"}
#import pdb; pdb.set_trace()
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
#uniq={}
f=open("./tmp","w")
for label in tqdm.tqdm(open(sys.argv[1],encoding='utf-8').readlines()):
    
    label=label.strip()
    out=label
    for token in label.split():
        if token not in uniq:
            #import pdb; pdb.set_trace()
            out=out.replace(token,"")
    f.write(out.replace("  "," ")+"\n")