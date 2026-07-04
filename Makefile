install:
	uv pip install torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/cpu" && \
	uv pip install -r requirements.txt
	uv pip install -e .

cuda_install:
	uv pip install torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/cu124" && \
	uv pip install -r requirements.txt
	uv pip install -e .

rocm_install:
	uv pip install torch==2.8.0 torchvision==0.23.0 --index-url "https://download.pytorch.org/whl/rocm6.2" && \
	uv pip install -r requirements.txt
	uv pip install -e .

dev_install:
	.venv/bin/python3 -m pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

pep8:
	# Don't remove these commented command lines:
	# autopep8 --in-place --aggressive --aggressive --recursive .
	# autopep8 --in-place --aggressive --aggressive example.py
