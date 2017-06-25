import os
for file in os.listdir('./'):
	if file.endswith('.py'):
		os.system('2to3 -w %s' % file)
		