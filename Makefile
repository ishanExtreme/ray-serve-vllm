ZIP_FILE = serve.zip
VLLM_VERSION=0.6.5
RAY_IMAGE_VERSION=2.40.0.22541c-py310-cu121
APP_DOCKER_IMAGE_VERSIONED = vllm-serve:latest

package-container:
	zip -r $(ZIP_FILE) . --exclude "venv/*" ".git/*" "*.pyc"
	docker build \
		-t $(APP_DOCKER_IMAGE_VERSIONED) \
		-f ./dockerfile.ray \
		--build-arg RAY_IMAGE_VERSION=$(RAY_IMAGE_VERSION) \
		--build-arg VLLM_VERSION=$(VLLM_VERSION) \
		.
	rm -f $(ZIP_FILE)
