# Makefile

# Run unit tests
test:
	pytest tests -v

# Format code using black
format:
	black src tests

# Run the full pipeline
run:
	python run_pipeline.py --data-root data/raw --output data/interim/players.csv

# Clean temporary or compiled files
clean:
	find . -name '*.pyc' -delete
