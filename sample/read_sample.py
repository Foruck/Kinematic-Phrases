import joblib
import numpy as np
meta = joblib.load('meta_info.pkl')
phrase = joblib.load('phrase.pkl')
motion = joblib.load('motion.pkl')
print(phrase.shape)
for key in meta['META2IDX'].keys():
    print(key, len(meta['META2IDX'][key]))
for i, info in enumerate(meta['IDX2META']):
    if i < 20:
        print(info, np.unique(phrase[:, i]))
print(motion.keys())