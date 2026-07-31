#!/usr/bin/env python
import MEGnet; 
import os,os.path as op;  
import shutil;
import sys;
replace_dir=op.join(MEGnet.__path__[0], "model_v2"); 
_approve='y'
print(replace_dir)
if op.exists(replace_dir): _approve=input(f"Do you want to replace the the model weights for MEGnet?(y/n)\n");
if _approve.lower().startswith("n"): print("Not replacing weights, exiting"), sys.exit();
if op.exists(replace_dir): shutil.rmtree(replace_dir);
newmodel=op.join(sys.argv[1], "model");
shutil.copytree(newmodel, replace_dir);
print("Successfully switched weights")
