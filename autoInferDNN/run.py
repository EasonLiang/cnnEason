#!/usr/bin/python

import os
import sys
from fnmatch import fnmatch

verStr=os.popen("head -1 /etc/issue | awk '{print $1,$2}'|sed -e 's/ //g;s/\./_/g'").read().split('\n')[0]
libPath='libs/'+verStr
sys.path.insert(0,libPath)

if verStr == 'ArchLinux' :
	import AutoInfer_cpp2python_ArchLinux as easonPy
elif verStr == 'Ubuntu16_04_7' :
	import AutoInfer_cpp2python_Ubuntu16_04_7 as easonPy
elif verStr == 'Ubuntu22_04_2' :
	import AutoInfer_cpp2python_Ubuntu22_04_2 as easonPy

def clean(object):
	cacheDir=libPath + '/__pycache__/'
	if os.path.exists(cacheDir):
		if os.path.exists(cacheDir + os.listdir(cacheDir)[0]):
			os.remove(cacheDir + os.listdir(cacheDir)[0])
		os.rmdir(cacheDir)

	for i in os.listdir(libPath):
		if fnmatch(i,'*.pyc'):
			os.remove( libPath + '/' + i)

	del object

obj = easonPy.Eason()
##obj.train(80000,False)

obj.setInput_TargetOutput([0.05,0.1],[0.13,0.99]);
#obj.train(8000,False)

obj.train(0,False,True,0.000027985)

clean(obj)
