import os
# rates=[0.03,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5]
rates=[0.1,0.2,0.4,0.5,0.6,0.8,1.0]
# rates=[1.0]

for rate in rates:
	os.system('python combine_reasonet_dump.py %f' %(rate))
	os.system('predict.bat')