# encrypted-traffic-classifier
Encrypted Network Traffic Classification using Deep Learning

project running commands: 
# build dataset
python src/preprocessing/build_dataset.py
# for checking the dataset and prediction 
python src/evaluation/check_pred_distribution.py
python src/evaluation/check_dataset.py
# for saving norms 
python src/preprocessing/save_norm_stats.py
# train model
python src/training/train_parallel_cnn_nin.py

# evaluate
python src/evaluation/evaluate_parallel_model.py

# run API
uvicorn api.main:app --reload

# run Streamlit
streamlit run streamlit_app.py


Classes:
0 → VPN
1 → Non-VPN
2 → TOR
