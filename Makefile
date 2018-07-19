init:
	pip install -r requirements.txt
	mkdir bin

test:
	pytest 

clean:
	@rm -f -r bin
	@rm -f -r tests/__pycache__
	@rm -f -r src/__pycache__
	@rm -f src/*.pyc
	@rm -f -r .cache/
	@rm -f -r .DS_Store/
	@rm -f -r .pytest_cache/

