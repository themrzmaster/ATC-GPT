PY ?= venv/bin/python
SERVER ?= http://127.0.0.1:5000

.PHONY: help install serve test e2e e2e-list e2e-server fake-xplane charts probe clean-cache

help:
	@echo "make install       - create venv + install requirements"
	@echo "make serve         - run the ATC server (http://127.0.0.1:5000)"
	@echo "make test          - offline unit + API tests (no API keys needed, <2s)"
	@echo "make e2e           - LIVE LLM scenarios in-process (pilot agent + judge). S=<name> C=<category> J=<parallel> R=<repeats>"
	@echo "make e2e-list      - list scenarios by category"
	@echo "make e2e-server    - same, but against the running server (SERVER=$(SERVER))"
	@echo "make fake-xplane   - drive the running server like the X-Plane plugin would (P=sbgr|kjfk)"
	@echo "make charts I=SBGR - list charts from the official providers"
	@echo "make probe I=SBGR  - one-shot ATC reply from the CLI (no server)"

install:
	python3 -m venv venv && venv/bin/pip install -r requirements.txt

serve:
	$(PY) atc_server.py

test:
	$(PY) -m pytest tests -q

e2e:
	$(PY) e2e/run_e2e.py $(if $(S),-s $(S),) $(if $(C),-c $(C),) -j $(or $(J),6) -r $(or $(R),1)

e2e-list:
	$(PY) e2e/run_e2e.py --list

e2e-server:
	$(PY) e2e/run_e2e.py --server $(SERVER) $(if $(S),-s $(S),)

fake-xplane:
	$(PY) e2e/fake_xplane.py --server $(SERVER) --profile $(or $(P),sbgr) $(if $(AUDIO),--audio,)

charts:
	$(PY) chart_providers.py $(or $(I),SBGR)

probe:
	$(PY) atc_llm_service.py $(or $(I),KJFK)

clean-cache:
	rm -rf .cache/charts
