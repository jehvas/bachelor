#SETUP
git clone http://github.com/jehavs/bachelor.git
cd bachelor
mkdir out
mkdir Result
mkdir data
mkdir data/20Newsgroups
mkdir data/SensitiveEnron
mkdir data/SensitiveEnron/evidence_tampering
mkdir data/SensitiveEnron/financial_state
mkdir data/SpamAssassin
mkdir data/Trustpilot
mkdir data/GloveWordEmbedding
pip install tensorflow==2.0.0-alpha0
pip install joblib
pip install matplotlib
pip install numpy
pip install pandas
pip install scikit-learn
pip install seaborn
pip install nltk
python3 -c "exec(\"import nltk\nltk.download('stopwords')\")

#Running
python main.py [algorithm] [dataset] [dataset size]

OR

python parameter_search.py [algorithm] [dataset] [dataset size] [num iteration]

OR

python memsafe_search.py [algorithm] [dataset] [dataset size] [num iteration]