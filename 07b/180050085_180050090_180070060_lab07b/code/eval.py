import torch
import pickle
import os

import training
if __name__ == "__main__":
	path = "../"

	device = "cpu"


	with open(os.path.join(os.path.join(path, "data"), "data_test.pkl"), 'rb') as f:
	    test_data = pickle.load(f)
	model = training.LiftModel(2, 1024, 0.5)
	model.load_state_dict(torch.load(os.path.join(os.path.join(path, "results"), 'bestModel.pt'), map_location = device))
	print("Test")
	training.run_epoch_weak(1,test_data, model, None, None,64, 'test')
