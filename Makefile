############################################## ghislain.bernard@gmail.com ##############################################

all: clean

SHELL:=/bin/bash

########################################################################################################################

clean:
	@echo ''
	@echo -e '/!\ cleaning...'

	rm --force --recursive --verbose __pycache__
	rm --force --recursive --verbose module/__pycache__

	@echo ...done

############################################## ghislain.bernard@gmail.com ##############################################
