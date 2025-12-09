#!/usr/bin/env bash
# Install dependencies first
pip install -r requirements.txt

# Run Streamlit
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
