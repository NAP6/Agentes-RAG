FROM downloads.unstructured.io/unstructured-io/unstructured:latest
LABEL authors="nicolas"

# Install Jupyter Notebook
RUN pip install jupyter

# Install Poppler using the package manager available in the base image
#RUN apk update && apk add poppler-utils

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Set the working directory
WORKDIR /project

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]