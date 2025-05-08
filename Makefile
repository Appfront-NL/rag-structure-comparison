.PHONY: install run clean

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && python comparison/test.py

clean:
	rm -rf .venv
