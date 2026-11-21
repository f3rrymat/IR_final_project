PY ?= python3
COLLECTIONS ?= cs scifact nfcorpus

.PHONY: setup data index calibrate-costs test test-web e2e eval-quick eval ablations web serve dev pool clean-index

setup:            ## install requirements (CPU torch) + the web UI's packages
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt
	cd web && npm install

data:             ## download sources, extract, clean, chunk -> data/processed/<collection>/passages.jsonl
	$(PY) -m contextweave.ingestion.download
	$(PY) scripts/get_beir.py
	$(PY) main.py data

index:            ## inverted index, champion lists, embeddings, FAISS (all collections)
	$(PY) main.py build-index --collection $(shell echo $(COLLECTIONS) | tr ' ' ',')

calibrate-costs:  ## measure method costs and update config/settings.yaml
	for c in $(COLLECTIONS); do $(PY) main.py calibrate-costs --collection $$c; done

test:             ## pytest (engine + API) and the web UI's unit tests
	$(PY) -m pytest
	cd web && npm test

test-web:         ## web UI unit tests + type check only
	cd web && npm run type-check && npm test

e2e:              ## browser tests against the running server (python main.py serve) - uses installed Edge
	cd web && npm run e2e

eval-quick:       ## tune on validation queries, then main systems on validation
	$(PY) experiments/fit_and_tune.py $(COLLECTIONS)
	$(PY) experiments/run_all.py $(COLLECTIONS) --split val --quick

eval:             ## all systems on test queries + ablations + sweeps + plots + tables
	$(PY) experiments/fit_and_tune.py $(COLLECTIONS)
	$(PY) experiments/run_all.py $(COLLECTIONS) --split test

pool:             ## build the judging pool for the cs benchmark
	$(PY) -c "import json; from contextweave.system import load_system; from contextweave.evaluation.benchmark import *; s=load_system(); json.dump(build_pool(s.engine(), load_queries('cs')), open('benchmark/pool.json','w'), indent=1)"

ablations:        ## ablations + quality-vs-cost sweep only
	$(PY) experiments/ablations.py $(COLLECTIONS)

web:              ## build the web UI into web/dist
	cd web && npm run build

serve:            ## UI + API at http://127.0.0.1:8000 (run `make web` once first)
	$(PY) main.py serve

dev:              ## UI with hot reload at http://localhost:5173 (needs `make serve` in another terminal)
	cd web && npm run dev

clean-index:
	rm -rf indexes/
