import os, pickle

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'models', 'manifest.pkl')

with open(MANIFEST_PATH, 'rb') as f:
    manifest = pickle.load(f)

manifest['lung_model_path']   = os.path.join(BASE_DIR, 'models', 'lung_model.pth')
manifest['bone_model_path']   = os.path.join(BASE_DIR, 'models', 'bone_model.pth')
manifest['router_model_path'] = os.path.join(BASE_DIR, 'models', 'router.pth')

with open(MANIFEST_PATH, 'wb') as f:
    pickle.dump(manifest, f)

print("Done!")
print("Lung  :", manifest['lung_model_path'])
print("Bone  :", manifest['bone_model_path'])
print("Router:", manifest['router_model_path'])