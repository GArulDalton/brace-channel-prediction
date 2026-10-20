import os, sys, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODELS = os.path.join(ROOT, 'models'); RESULTS = os.path.join(ROOT, 'results')
os.makedirs(MODELS, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)


def model_path(channel, domain, tau):
    return os.path.join(MODELS, f'{channel}_{domain}_tau{tau}.pkl')


def load_model(channel, domain, tau):
    with open(model_path(channel, domain, tau), 'rb') as f:
        return pickle.load(f)


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
