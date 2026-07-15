.PHONY: install install-lite sample build dashboard clean

install:
	pip install -r requirements.txt

install-lite:
	pip install -r requirements-lite.txt

sample:
	python -m scripts.generate_sample_data

build:
	python -m src.build_dataset

dashboard:
	streamlit run app/streamlit_app.py

collect-trends:
	python -m src.collect_trends

collect-reddit:
	python -m src.collect_reddit

collect-news:
	python -m src.collect_news

collect-all: collect-reddit collect-news collect-trends
	@echo "All collectors finished. Now run 'make build' to rebuild processed tables."

clean:
	rm -rf data/processed/*.parquet
