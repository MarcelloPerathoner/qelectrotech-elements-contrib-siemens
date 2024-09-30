BUILD_DIR=./build
ELEM_DIR=$(HOME)/.qet/elements

SCRAPE_DIR=$(BUILD_DIR)/scrape
LIB_DIR=$(ELEM_DIR)/siemens

LIB=$(BUILD_DIR)/lib/siemens.tar.gz

PYTHON=./.venv/bin/python

lib: $(LIB)

$(LIB): $(SCRAPE_DIR)/*.txt
	mkdir -p "$(LIB_DIR)"
	$(PYTHON) -m src.generator $^
	mkdir -p $(@D)
	tar cjf $(LIB) -C $(ELEM_DIR) siemens

init:
	python -m venv .venv --system-site-packages
	source .venv/bin/activate
	pip install -r requirements.txt

scrape:
	mkdir -p $(SCRAPE_DIR)
	$(PYTHON) -m src.scraper $(SCRAPE_DIR)

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)
